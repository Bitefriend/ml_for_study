# =========================
# AutoGluon GPU 학습/제출 스크립트 (Windows / RTX 5080)
# 규정 메모: 외부데이터 금지, 제출 스키마(ID, probability), ROC-AUC
# =========================

import os
import pandas as pd
from uuid import uuid4
from datetime import datetime
from autogluon.multimodal import MultiModalPredictor

# [ADD] NVML DLL 경로 보장 (pynvml 로딩 오류 방지)
NVML_CANDIDATES = [
    r"C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll",
    r"C:\Windows\System32\nvml.dll",
]
for _p in NVML_CANDIDATES:
    if os.path.exists(_p):
        os.environ.setdefault("NVML_DLL", _p)
        try:
            os.add_dll_directory(os.path.dirname(_p))  # Python 3.8+ DLL 검색 경로 추가
        except Exception:
            pass
        break

def make_unique_run_dir(base_dir: str) -> str:
    """[CHG] 경로 충돌 방지: 원자적으로 유니크 폴더 생성."""
    import tempfile
    os.makedirs(base_dir, exist_ok=True)  # outputs 생성만 허용
    run_dir = tempfile.mkdtemp(prefix="ag_run_예시_", dir=base_dir)  # (예시 — 추후 작성 예정)
    return run_dir

def to_positive_prob(proba_out):
    """[ADD] predict_proba 반환 형식의 차이를 흡수하여 양성확률 벡터로 변환."""
    if hasattr(proba_out, "ndim") and proba_out.ndim == 1:  # Series
        return proba_out.values
    if hasattr(proba_out, "columns"):                       # DataFrame
        for key in (1, "1", "positive", "True", True):
            if key in proba_out.columns:
                return proba_out[key].values
        return proba_out.iloc[:, -1].values                 # 폴백
    import numpy as np
    arr = getattr(proba_out, "values", proba_out)
    arr = np.asarray(arr)
    return arr[:, -1] if arr.ndim == 2 else arr

def main():
    # [ADD] Windows 멀티프로세싱 안전장치(명시적 설정은 선택)
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # -----------------------
    # 1) 데이터 로드 & 타입 고정
    # -----------------------
    train = pd.read_csv("data/raw/train.csv")   # [ID, URL, label]
    test  = pd.read_csv("data/raw/test.csv")    # [ID, URL]

    # [CHG] 텍스트/라벨 타입 명시
    train_ag = train[['URL', 'label']].copy()
    train_ag['text']  = train_ag['URL'].astype(str)
    train_ag['label'] = train_ag['label'].astype(int)
    train_ag = train_ag[['text', 'label']]

    test_ag = test[['ID', 'URL']].copy()
    test_ag['text'] = test_ag['URL'].astype(str)
    test_ag = test_ag[['ID', 'text']]

    print("[INFO] dtypes:\n", train_ag.dtypes)
    print("[INFO] head:\n", train_ag.head(3))

    # -----------------------
    # 2) 고유 저장 경로 생성 (충돌 방지)
    # -----------------------
    BASE_DIR = "outputs"                              # (예시 — 추후 작성 예정)
    os.makedirs(BASE_DIR, exist_ok=True)
    RUN_DIR = os.path.join(
        BASE_DIR,
        f"ag_run_예시_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{os.getpid()}_{uuid4().hex[:6]}"
    )
    os.makedirs(RUN_DIR, exist_ok=False)
    print("[INFO] Save/Logs dir:", RUN_DIR)

    # -----------------------
    # 3) 모델 생성 & 학습
    # -----------------------
    predictor = MultiModalPredictor(
        label='label',
        problem_type='binary',
        eval_metric='roc_auc',
        path=RUN_DIR,    # ← 고정 문자열 절대 금지!
    )

    fit_kwargs = dict(
        train_data=train_ag,
        time_limit=3600,            # (예시 — 추후 작성 예정)
        column_types={"text": "text"},        # [ADD] 텍스트 컬럼 강제 지정
        hyperparameters={
            "env.num_gpus": 1,                # GPU 1개 (가용 전부는 -1)
            "model.hf_text.checkpoint_name": "kmack/malicious-url-detection",
            # 나머지 하이퍼파라미터는 전부 **(예시 — 추후 작성 예정)**
        }
    )

    print("[INFO] Training start …")
    predictor.fit(**fit_kwargs)

    # -----------------------
    # 4) 추론 & 제출 파일 생성
    # -----------------------
    proba = predictor.predict_proba(test_ag)
    p = to_positive_prob(proba)
    submission = pd.DataFrame({"ID": test_ag["ID"], "probability": p})

    sub_name = f"submission_autogluon_hf_예시_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    SUB_PATH = os.path.join(RUN_DIR, sub_name)
    submission.to_csv(SUB_PATH, index=False)

    print("[DONE] Saved submission:", SUB_PATH)
    print("[CHECK] shape:", submission.shape, "| columns:", list(submission.columns))
    print("[CHECK] NaN:", submission.isna().sum().to_dict())

# [ADD] Windows 멀티프로세싱 재실행 방지: 메인 가드
if __name__ == "__main__":
    main()
