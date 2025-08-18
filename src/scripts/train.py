import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

import lightgbm as lgb
import xgboost as xgb

# --- 데이터 로드 (열 이름은 사용자 데이터에 맞게 교체 필요) ---
# train.csv: [ID, URL, label] / test.csv: [ID, URL] 
train = pd.read_csv("data/raw/train.csv")  
test  = pd.read_csv("data/raw/test.csv")   
X = train["URL"]                           
y = train["label"] 

# --- 벡터라이저 (문자 3–5 gram, 해시 피처 수는 예시값) ---
# vec = HashingVectorizer(analyzer='char', ngram_range=(3,5),
#                         n_features=2**18, alternate_sign=False, lowercase=False)

# -- 벡터라이저들 (모델 성향에 맞춰 분리) --
vec_hash_35 = HashingVectorizer(        # 선형 모델용 (희박·초고차원에 강함)
    analyzer='char', ngram_range=(3, 5),
    n_features=2**18, alternate_sign=False, lowercase=False
)
vec_tfidf_linear = TfidfVectorizer(     # 선형 모델 대안
    analyzer='char', ngram_range=(3, 5),
    sublinear_tf=True, lowercase=False, max_features=None
)
vec_tfidf_tree = TfidfVectorizer(       # 트리 계열용 (차원 제한 권장)
    analyzer='char', ngram_range=(3, 4),
    sublinear_tf=True, lowercase=False, max_features=50000
)
svd_300 = TruncatedSVD(                 # (선택) 트리계열 안정화용 차원 축소
    n_components=300, random_state=4321
)

print("X dtype:", X.dtype, " / sample:", X.iloc[:3].tolist())
print("y dtype:", y.dtype, " / counts:\n", y.value_counts())
print((y.astype(str) + " 인식").iloc[:5].tolist())


# --- 모델 정의 (이 부분만 추가) ---

# 1. 로지스틱 회귀 모델 (기본 모델)
log_reg_model = LogisticRegression(solver='liblinear', random_state=42)
# 권장 추가: penalty='l2' or 'elasticnet', C=**예시**, max_iter=**예시**, class_weight='balanced'
print("\n로지스틱 회귀 모델 정의 완료.")

# 2. 선형 SVM 모델 (확률 보정이 필요)
svc_model = CalibratedClassifierCV(LinearSVC(random_state=42))
 # 권장 추가: C=**예시**, loss='squared_hinge', max_iter=**예시**, tol=**예시**, class_weight='balanced'
print("선형 SVM 모델 정의 완료.")

# 3. 랜덤 포레스트 모델 (앙상블 모델)
rf_model = RandomForestClassifier(random_state=42)
    # 권장 추가: n_estimators=**예시**, max_depth=**예시**, min_samples_split=**예시**,
    #           min_samples_leaf=**예시**, max_features='sqrt', class_weight='balanced_subsample', n_jobs=-1
print("랜덤 포레스트 모델 정의 완료.")

# 4. LightGBM 모델 (부스팅 계열, 설치 후 사용)
lgbm_model = lgb.LGBMClassifier(random_state=42)
    # 권장 추가: n_estimators=**예시**, learning_rate=**예시**, num_leaves=**예시**, max_depth=**예시**,
    #           feature_fraction=**예시**, bagging_fraction=**예시**, bagging_freq=**예시**,
    #           min_data_in_leaf=**예시**, lambda_l1=**예시**, lambda_l2=**예시**,
    #           objective='binary', metric='auc', is_unbalance=True, num_threads=-1
print("LightGBM 모델 정의 완료.")

# 5. XGBoost 모델 (부스팅 계열, 설치 후 사용)
xgb_model = xgb.XGBClassifier(random_state=42)
# 권장 추가: n_estimators=**예시**, learning_rate=**예시**, max_depth=**예시**, min_child_weight=**예시**,
    #           subsample=**예시**, colsample_bytree=**예시**, reg_alpha=**예시**, reg_lambda=**예시**,
    #           gamma=**예시**, tree_method='hist', eval_metric='auc', n_jobs=-1
print("XGBoost 모델 정의 완료.")


# -- 파이프라인 (벡터라이저까지 캡슐화: 누수 방지) --
pipe_log_reg_model = Pipeline([("vec", vec_hash_35),      ("clf", log_reg_model)])     # 또는 vec_tfidf_linear
pipe_svc_model     = Pipeline([("vec", vec_hash_35),      ("clf", svc_model)])         # 확률 출력 OK
pipe_rf_model      = Pipeline([("vec", vec_tfidf_tree),   ("svd", svd_300), ("clf", rf_model)])  # 트리+SVD 권장
pipe_lgbm_model    = Pipeline([("vec", vec_tfidf_tree),   ("clf", lgbm_model)])        # LGBM은 SVD 생략 예시
pipe_xgb_model     = Pipeline([("vec", vec_tfidf_tree),   ("svd", svd_300), ("clf", xgb_model)])

# 한 번에 돌리기 쉽게 딕셔너리로 관리
pipelines = {
    "log_reg_model": pipe_log_reg_model,
    "svc_model":     pipe_svc_model,
    "rf_model":      pipe_rf_model,
    "lgbm_model":    pipe_lgbm_model,
    "xgb_model":     pipe_xgb_model,
}