GPU 세팅

목표: Windows + RTX 5080 환경에서 XGBoost/LightGBM 학습을 GPU로 실행·재현.
최종 확인된 조합: Python 3.11.13 / XGBoost 3.0.4 / LightGBM 4.6.0 / NVIDIA Driver 572.83 / CUDA 12.8

개요

우리가 겪은 이슈들을 쭉 해결해 오면서 드라이버 → CUDA 런타임 → Conda 환경 → 패키지 → NVML → XGBoost/LightGBM 설정까지 한 번에 정리했다.

특히 XGBoost 2.x/3.x에서는 device='cuda'만 쓰고 gpu_id를 같이 주면 에러가 난다는 점이 핵심 포인트.

단계별 가이드
1) 드라이버 & CUDA 런타임 설치/확인

NVIDIA 드라이버 설치(또는 최신 유지)

RTX 5080 기준 확인값: Driver 572.83, CUDA Runtime 12.8.

확인

nvidia-smi


GPU 이름, Driver 버전, CUDA Version(12.8)을 확인한다.

메모: XGBoost/LightGBM pip wheel은 내부에 필요한 런타임을 포함한다. 별도의 “CUDA Toolkit” 설치는 필수 아님(드라이버만 정확히 설치되면 충분).

2) Conda 환경 준비

아래 환경명은 실제 우리가 쓰던 이름 그대로 적음.

# (이미 만들어져 있으면 생략)
conda create -n project_computerVision python=3.11 -y

# 활성화
conda activate project_computerVision

# 파이썬/핍 경로가 해당 환경을 가리키는지 확인
where python
where pip
python -c "import sys; print(sys.executable)"

3) 패키지 설치 (CUDA 대응 버전)
# 핵심 패키지
pip install -U xgboost==3.0.4 lightgbm==4.6.0 pynvml

# (옵션) PyTorch CUDA 12.8 빌드 — 프로젝트 필수는 아님
# pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio

# (일반 라이브러리 — 이미 있을 수 있음)
pip install -U numpy pandas scikit-learn


버전은 우리 환경에서 최종 동작이 확인된 값이다(3.0.4 / 4.6.0). 팀 환경이 다르면 최신으로 올려도 되지만, 재현 목적이면 위 버전 고정 추천.

4) NVML(모니터링 라이브러리) 인식 문제 해결

증상: pynvml.NVMLError_LibraryNotFound: NVML Shared Library Not Found

해결(둘 중 하나면 충분)

A. 환경변수로 NVML 경로 지정

setx NVML_DLL "C:\Windows\System32\nvml.dll"
# 현재 PowerShell 세션 한정
$env:NVML_DLL = "C:\Windows\System32\nvml.dll"


B. 표준 폴더에 nvml.dll 복사

mkdir "C:\Program Files\NVIDIA Corporation\NVSMI" 2>nul
copy /Y "C:\Windows\System32\nvml.dll" "C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll"


검증

python - << 'PY'
import pynvml
pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(0)
print("GPU:", pynvml.nvmlDeviceGetName(h))
pynvml.nvmlShutdown()
PY


→ GPU: b'NVIDIA GeForce RTX 5080'가 나오면 OK.

5) LightGBM(GPU) 사전 체크 (선택)

pip wheel(4.6.0) 기준, OpenCL GPU 경로가 포함되어 바로 동작하는 경우가 많다.

처음 학습 시 콘솔에 dep-*.d 관련 경고가 여러 줄 뜰 수 있는데, OpenCL 커널 캐시 생성 로그로 무해하다.

만약 환경에 따라 GPU가 불안정하면, 이번 프로젝트는 **XGBoost(GPU)**만으로도 충분 → LightGBM은 선택 사항으로 두자.

6) XGBoost(GPU) 사전 체크 (필수)

중요 규칙 (2.x/3.x)

device='cuda', tree_method='hist' 사용

gpu_id를 같이 주면 안 됨 → Both device and gpu_id are specified. Use device instead. 에러

미니 테스트

import numpy as np, xgboost as xgb
from packaging import version

X = np.random.RandomState(0).rand(32,4).astype(np.float32)
y = np.random.RandomState(0).randint(0,2,32)
d = xgb.DMatrix(X, label=y)
params = {"max_depth":1, "eta":1.0, "objective":"binary:logistic", "nthread":1}
if version.parse(xgb.__version__) >= version.parse("2.0.0"):
    params.update(device="cuda", tree_method="hist")   # gpu_id 금지
else:
    params.update(tree_method="gpu_hist", predictor="gpu_predictor", gpu_id=0)

xgb.train(params, d, num_boost_round=1)
print("OK")

7) 학습 파이프라인 설계 팁 (CPU 최적화 + GPU 체감)

전처리는 HashingVectorizer(3~5-gram) 로 한 번만 수행 후 K-Fold에서 재사용
→ 전처리 CPU 시간을 대폭 줄여 GPU 학습 구간이 명확히 보임

XGBoost 학습 시 CPU 스레드 제한: nthread=1

XGBoost 목적함수는 binary:logistic (LightGBM식 binary와 혼동 금지)

검증용 최소 실행 스크립트 (Fold별 GPU 사용률 출력)

파일 경로는 팀 리포지토리에 맞게 배치. 아래는 동작 확인된 스크립트.

import os, sys, time, threading
import numpy as np, pandas as pd
from packaging import version
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import roc_auc_score
import xgboost as xgb

# NVML hint(Windows)
for p in [r"C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll", r"C:\Windows\System32\nvml.dll"]:
    if os.path.exists(p):
        os.environ.setdefault("NVML_DLL", p)
        try: os.add_dll_directory(os.path.dirname(p))
        except: pass
        break

def xgb_gpu_ok():
    try:
        X = np.random.RandomState(0).rand(32,4).astype(np.float32)
        y = np.random.RandomState(0).randint(0,2,32)
        d = xgb.DMatrix(X, label=y)
        params = {"max_depth":1, "eta":1.0, "objective":"binary:logistic", "nthread":1}
        if version.parse(xgb.__version__) >= version.parse("2.0.0"):
            params.update(device="cuda", tree_method="hist")
        else:
            params.update(tree_method="gpu_hist", predictor="gpu_predictor", gpu_id=0)
        xgb.train(params, d, num_boost_round=1); return True
    except Exception as e:
        print("[CHK] XGBoost GPU fail:", e); return False

class GPUMonitor:
    def __init__(self, index=0, interval=0.5):
        self.index=index; self.interval=interval; self.samples=[]; self.enabled=False; self._stop=False
    def start(self):
        try:
            import pynvml; self.nv=pynvml; self.nv.nvmlInit()
            self.h=self.nv.nvmlDeviceGetHandleByIndex(self.index); self.enabled=True
            def run():
                import time
                while not self._stop:
                    u=self.nv.nvmlDeviceGetUtilizationRates(self.h)
                    m=self.nv.nvmlDeviceGetMemoryInfo(self.h)
                    self.samples.append((u.gpu, u.memory, m.used, m.total)); time.sleep(self.interval)
            threading.Thread(target=run, daemon=True).start()
        except Exception as e: print("[MON] disabled:", e)
    def stop(self, prefix=""):
        if not self.enabled: return
        self._stop=True; self.nv.nvmlShutdown(); self.summary(prefix)
    def summary(self, prefix=""):
        if not self.samples: print(prefix+"[MON] no samples"); return
        import numpy as np
        a=np.asarray(self.samples,float); g,m,u,t=a[:,0],a[:,1],a[:,2],a[:,3][0]
        print(prefix+f"[MON] GPU% avg/max {g.mean():.1f}/{g.max():.1f} | Mem% {m.mean():.1f}/{m.max():.1f} | "
                     f"MemGB {u.mean()/1e9:.2f}/{u.max()/1e9:.2f} of {t/1e9:.2f}")

def main():
    assert xgb_gpu_ok(), "XGBoost GPU가 동작하지 않습니다."
    train = pd.read_csv("data/raw/train.csv")  # [ID, URL, label]
    X_text = train["URL"].fillna("").astype(str)
    y = train["label"].astype(int)

    vec = HashingVectorizer(analyzer="char", ngram_range=(3,5),
                            n_features=2**18, alternate_sign=False, lowercase=False, dtype=np.float32)
    X_all = vec.transform(X_text)

    xgb_params = {"random_state":42, "eval_metric":"auc", "objective":"binary:logistic", "nthread":1}
    if version.parse(xgb.__version__) >= version.parse("2.0.0"):
        xgb_params.update(device="cuda", tree_method="hist")
    else:
        xgb_params.update(tree_method="gpu_hist", predictor="gpu_predictor", gpu_id=0)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_p, oof_y = [], []
    for f,(tr,va) in enumerate(skf.split(X_all, y.values),1):
        print(f"[RUN] Fold {f}/5 (train={len(tr):,}, valid={len(va):,})")
        model = xgb.XGBClassifier(**xgb_params)
        mon = GPUMonitor(0, 0.5); mon.start()
        t0=time.time(); model.fit(X_all[tr], y.values[tr]); sec=time.time()-t0
        p = model.predict_proba(X_all[va])[:,1]
        mon.stop(prefix=f"[F{f}] "); print(f"[F{f}] fit time: {sec:.1f}s")
        oof_p.append(p); oof_y.append(y.values[va])

    from numpy import concatenate as cat
    print("OOF ROC-AUC:", roc_auc_score(cat(oof_y), cat(oof_p)))

if __name__=="__main__":
    main()

문제 → 원인 → 해결 요약
증상/로그	원인	해결
NVMLError_LibraryNotFound	Python이 nvml.dll을 못 찾음	NVML_DLL 환경변수 지정 또는 nvml.dll을 NVSMI 폴더에 복사
Both device and gpu_id are specified	XGBoost 2.x/3.x에서 device='cuda'와 gpu_id 동시 지정	gpu_id 제거 (필요 시 CUDA_VISIBLE_DEVICES로 GPU 선택)
Objective candidate …	XGBoost에 LightGBM식 binary 전달	objective='binary:logistic' 로 교체
n_features=2 18 SyntaxError	오타	n_features=2**18 로 수정
LightGBM dep-*.d 경고 다수	OpenCL 커널 초기 빌드 로그	무해, 무시 가능(한 번 빌드되면 감소)
체크리스트

 nvidia-smi로 Driver/CUDA 확인(572.83 / 12.8)

 Conda env 내부 python/pip로 동작 확인

 pynvml GPU 이름 정상 조회

 XGBoost 미니 테스트 OK

 검증 스크립트에서 Fold별 GPU%/Mem% 요약 출력

 (제출용) CSV 스키마: ID, probability

실행 커맨드 예시
# 학습/검증
python src/scripts/train_for_gpu_test.py

# (선택) 특정 GPU만 사용
set CUDA_VISIBLE_DEVICES=0 & python src/scripts/train_for_gpu_test.py

다음 행동

리포지토리에 이 문서를 docs/GPU_세팅.md로 커밋.

최종 하이퍼파라미터/실행 커맨드는 프로젝트 README에 추적.

LightGBM(GPU)을 쓰고 싶다면 LGBMClassifier(device='gpu')로 시도하되, 불안정 시 XGBoost(GPU)로 고정.
