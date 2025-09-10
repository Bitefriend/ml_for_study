# XGBoost(GPU) 5-Fold + GPU 사용률 모니터 + GPU 재시도(메모리 세이프)
# data/raw/train.csv  [ID, URL, label]

import os, sys, time, threading
import numpy as np
import pandas as pd
from packaging import version
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import roc_auc_score
import xgboost as xgb

# =========================
# 0) NVML 경로 힌트(Windows)
# =========================
NVML_CANDIDATES = [
    r"C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll",
    r"C:\Windows\System32\nvml.dll",
]
for pth in NVML_CANDIDATES:
    if os.path.exists(pth):
        os.environ.setdefault("NVML_DLL", pth)
        try:
            os.add_dll_directory(os.path.dirname(pth))
        except Exception:
            pass
        break

# =========================
# 1) 환경 로그 & GPU 체크
# =========================
def log_env():
    print(f"[ENV] Python={sys.version.split()[0]}  XGBoost={xgb.__version__}")
    try:
        import pynvml
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            gpus.append(name.decode() if isinstance(name, bytes) else str(name))
        print(f"[ENV] NVML GPUs: {gpus}")
        pynvml.nvmlShutdown()
    except Exception as e:
        print(f"[ENV] NVML check skipped ({type(e).__name__}: {e})")

def xgb_gpu_ok() -> bool:
    """아주 작은 학습으로 CUDA 가용성 빠르게 확인"""
    try:
        X = np.random.RandomState(0).rand(32, 4).astype(np.float32)
        y = np.random.RandomState(0).randint(0, 2, 32)
        dtrain = xgb.DMatrix(X, label=y)
        params = {"max_depth": 1, "eta": 1.0, "objective": "binary:logistic", "nthread": 1}
        if version.parse(xgb.__version__) >= version.parse("2.0.0"):
            params.update(device="cuda", tree_method="hist")
        else:
            params.update(tree_method="gpu_hist", predictor="gpu_predictor", gpu_id=0)
        xgb.train(params, dtrain, num_boost_round=1)
        print("[CHK] XGBoost(GPU) OK")
        return True
    except Exception as e:
        print(f"[CHK] XGBoost GPU not available ({type(e).__name__}: {e})")
        return False

# =========================
# 2) GPU 사용률 모니터(선택)
# =========================
class GPUMonitor:
    def __init__(self, index: int = 0, interval: float = 0.5):
        self.index = index
        self.interval = interval
        self.samples = []
        self._stop = False
        self.enabled = False
    def start(self):
        try:
            import pynvml
            self.pynvml = pynvml
            self.pynvml.nvmlInit()
            self.handle = self.pynvml.nvmlDeviceGetHandleByIndex(self.index)
            self.enabled = True
            def _run():
                while not self._stop:
                    util = self.pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                    mem = self.pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                    self.samples.append((time.time(), util.gpu, util.memory, mem.used, mem.total))
                    time.sleep(self.interval)
            self.t = threading.Thread(target=_run, daemon=True)
            self.t.start()
        except Exception as e:
            print(f"[MON] disabled ({type(e).__name__}: {e})")
    def stop(self, prefix=""):
        if not self.enabled: return
        self._stop = True
        self.t.join()
        self.pynvml.nvmlShutdown()
        self.summary(prefix)
    def summary(self, prefix=""):
        if not self.samples:
            print(prefix + "[MON] no samples"); return
        arr = np.asarray(self.samples, dtype=np.float64)
        gpu, memp, used, total = arr[:,1], arr[:,2], arr[:,3], arr[:,4][0]
        print(prefix + f"[MON] GPU% avg/max: {gpu.mean():.1f}/{gpu.max():.1f} | "
              f"Mem% avg/max: {memp.mean():.1f}/{memp.max():.1f} | "
              f"Mem(GB) avg/max: {used.mean()/1e9:.2f}/{used.max()/1e9:.2f} of {total/1e9:.2f}")

# =========================
# 3) 학습 본문
# =========================
def main():
    log_env()
    assert xgb_gpu_ok(), "XGBoost GPU가 동작하지 않습니다."

    # ---- 데이터 로드 ----
    train = pd.read_csv("data/raw/train.csv")  # [ID, URL, label]
    X_text = train["URL"].fillna("").astype(str)
    y = train["label"].astype(int)
    print("y counts:\n", y.value_counts())

    # ---- 해싱 전처리 (차원/메모리 절약 설정) ----
    # 16GB VRAM 환경 기준으로 안전한 기본값. 필요 시 더 줄여도 됨.
    N_FEATURES = 2**14          # 16384  (더 타이트하게: 2**13=8192)
    NGRAM = (3, 3)              # (3,4)가 무거우면 (3,3)로
    vec = HashingVectorizer(
        analyzer="char",
        ngram_range=NGRAM,
        n_features=N_FEATURES,
        alternate_sign=False,
        lowercase=False,
        binary=True,
        dtype=np.float32,
    )
    X_all = vec.transform(X_text)
    print("[INFO] Hashed features shape:", X_all.shape, "| dtype:", X_all.dtype)

    # ---- 공통 상수: max_bin 일관 유지 (QDM + Booster 모두 동일) ----
    # 너무 크면 VRAM 증가. 32~64 권장.
    MAX_BIN = 32

    # ---- 공통(XGBoost 2.x 기준) 기본 파라미터 (device/cuda, hist) ----
    BASE = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.1,
        "nthread": 1,
        "random_state": 42,
    }
    if version.parse(xgb.__version__) >= version.parse("2.0.0"):
        BASE.update(device="cuda", tree_method="hist")
    else:
        BASE.update(tree_method="gpu_hist", predictor="gpu_predictor", gpu_id=0)

    print("[DBG] Base params:", BASE, "| MAX_BIN:", MAX_BIN)

    # ---- GPU 재시도 헬퍼: 단계적으로 더 보수적인 설정으로 시도 ----
    def try_gpu_train(Xtr, ytr, Xva, yva):
        trials = [
            dict(max_bin=64, max_depth=5, subsample=0.6, colsample_bytree=0.15, min_child_weight=2),
            dict(max_bin=32, max_depth=4, subsample=0.5, colsample_bytree=0.10, min_child_weight=5),
            dict(max_bin=32, max_depth=4, subsample=0.4, colsample_bytree=0.10, min_child_weight=10, reg_lambda=2.0),
        ]
        for t in trials:
            params = dict(BASE, **t)
            try:
                dtrain = xgb.QuantileDMatrix(Xtr, label=ytr, max_bin=params["max_bin"])
                dvalid = xgb.QuantileDMatrix(Xva, label=yva, ref=dtrain, max_bin=params["max_bin"])
                mon = GPUMonitor(index=0, interval=0.5); mon.start()
                t0 = time.time()
                bst = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=300,
                    evals=[(dvalid, "valid")],
                    early_stopping_rounds=30,
                    verbose_eval=50,
                )
                sec = time.time() - t0
                preds = bst.inplace_predict(Xva)
                mon.stop(prefix=f"[GPU] ")
                print(f"[GPU] used params={t} | time={sec:.1f}s")
                return preds, params, "gpu"
            except xgb.core.XGBoostError as e:
                if "cudaErrorMemoryAllocation" in str(e) or "bad allocation" in str(e):
                    print("[RETRY] GPU OOM → 더 보수적인 설정으로 재시도:", t)
                    continue
                else:
                    raise

        # ---- 모든 GPU 시도 실패: CPU 폴백 ----
        cpu_params = dict(BASE)
        cpu_params.update(device="cpu", max_bin=32, max_depth=4,
                          subsample=0.5, colsample_bytree=0.10,
                          min_child_weight=5, reg_lambda=2.0)
        dtrain = xgb.DMatrix(Xtr, label=ytr)
        dvalid = xgb.DMatrix(Xva, label=yva)
        t0 = time.time()
        bst = xgb.train(
            cpu_params,
            dtrain,
            num_boost_round=200,
            evals=[(dvalid, "valid")],
            early_stopping_rounds=20,
            verbose_eval=50,
        )
        sec = time.time() - t0
        preds = bst.predict(dvalid)
        print(f"[CPU] used params={cpu_params} | time={sec:.1f}s")
        return preds, cpu_params, "cpu"

    # ---- 5-Fold K-Fold ----
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds, oof_y = [], []
    print("\n--- XGBoost 5-Fold (GPU 우선, 재시도 내장) ---")
    for fold, (tr, va) in enumerate(skf.split(X_all, y.values), 1):
        print(f"[RUN] Fold {fold}/5  (train={len(tr):,}, valid={len(va):,})")
        Xtr, Xva = X_all[tr], X_all[va]
        ytr, yva = y.values[tr], y.values[va]

        try:
            preds, used_params, mode = try_gpu_train(Xtr, ytr, Xva, yva)
        except xgb.core.XGBoostError as e:
            # GPU 외 예외는 그대로 노출
            raise

        oof_preds.append(preds)
        oof_y.append(yva)

    from numpy import concatenate as cat
    auc = roc_auc_score(cat(oof_y), cat(oof_preds))
    print(f"\n--- 결과 요약 ---\nOOF ROC-AUC: {auc:.4f}")

if __name__ == "__main__":
    main()
