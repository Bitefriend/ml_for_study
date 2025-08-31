# XGBoost(GPU) 5-Fold + GPU 사용률 모니터 (QuantileDMatrix ref + max_bin 일치)
# data/raw/train.csv  [ID, URL, label]

import os, sys, time, threading
import numpy as np
import pandas as pd
from packaging import version
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import roc_auc_score
import xgboost as xgb

# --- NVML 경로 힌트(Windows) ---
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

def main():
    log_env()
    assert xgb_gpu_ok(), "XGBoost GPU가 동작하지 않습니다."

    # ---- 데이터 로드 ----
    train = pd.read_csv("data/raw/train.csv")  # [ID, URL, label]
    X_text = train["URL"].fillna("").astype(str)
    y = train["label"].astype(int)
    print("y counts:\n", y.value_counts())

    # ---- 해싱 전처리 (차원 축소 & 이진화) ----
    vec = HashingVectorizer(
        analyzer="char",
        ngram_range=(3, 4),
        n_features=2**15,         # 32,768
        alternate_sign=False,
        lowercase=False,
        binary=True,
        dtype=np.float32,
    )
    X_all = vec.transform(X_text)
    print("[INFO] Hashed features shape:", X_all.shape, "| dtype:", X_all.dtype)

    # ---- 공통 상수: max_bin 일관 유지 ----
    MAX_BIN = 64  # 필요시 32/128로 조정

    # ---- XGBoost 파라미터 (GPU 저메모리 세팅) ----
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.6,
        "colsample_bytree": 0.2,
        "max_bin": MAX_BIN,       # ★ 부스터 파라미터
        "nthread": 1,
        "random_state": 42,
    }
    if version.parse(xgb.__version__) >= version.parse("2.0.0"):
        xgb_params.update(device="cuda", tree_method="hist")
    else:
        xgb_params.update(tree_method="gpu_hist", predictor="gpu_predictor", gpu_id=0)
    print("[DBG] XGB params:", xgb_params)

    # ---- 5-Fold + GPU 모니터 + QuantileDMatrix(ref + max_bin 일치) ----
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds, oof_y = [], []
    print("\n--- XGBoost(GPU) 5-Fold 시작 ---")
    for fold, (tr, va) in enumerate(skf.split(X_all, y.values), 1):
        print(f"[RUN] Fold {fold}/5  (train={len(tr):,}, valid={len(va):,})")
        Xtr, Xva = X_all[tr], X_all[va]
        ytr, yva = y.values[tr], y.values[va]

        try:
            # ★★ 핵심: QuantileDMatrix에도 동일한 max_bin 명시 + dvalid는 ref=dtrain
            dtrain = xgb.QuantileDMatrix(Xtr, label=ytr, max_bin=MAX_BIN)
            dvalid = xgb.QuantileDMatrix(Xva, label=yva, ref=dtrain, max_bin=MAX_BIN)

            mon = GPUMonitor(index=0, interval=0.5); mon.start()
            bst = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=300,
                evals=[(dvalid, "valid")],
                early_stopping_rounds=30,
                verbose_eval=50,
            )
            p = bst.inplace_predict(Xva)  # 복사 없이 예측
            mon.stop(prefix=f"[F{fold}] ")

        except xgb.core.XGBoostError as e:
            # GPU OOM 등 → CPU 폴백
            if "cudaErrorMemoryAllocation" in str(e) or "bad allocation" in str(e):
                print(f"[WARN][F{fold}] GPU OOM → CPU hist로 폴백합니다.")
                cpu_params = dict(xgb_params); cpu_params.update(device="cpu")
                dtrain = xgb.DMatrix(Xtr, label=ytr)
                dvalid = xgb.DMatrix(Xva, label=yva)
                bst = xgb.train(
                    cpu_params,
                    dtrain,
                    num_boost_round=200,
                    evals=[(dvalid, "valid")],
                    early_stopping_rounds=20,
                    verbose_eval=50,
                )
                p = bst.predict(dvalid)
            else:
                raise

        oof_preds.append(p); oof_y.append(yva)

    from numpy import concatenate as cat
    auc = roc_auc_score(cat(oof_y), cat(oof_preds))
    print(f"\n--- 결과 요약 ---\nXGBoost OOF ROC-AUC: {auc:.4f}")

if __name__ == "__main__":
    main()
