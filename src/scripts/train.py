import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
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
vec = HashingVectorizer(analyzer='char', ngram_range=(3,5),
                        n_features=2**18, alternate_sign=False, lowercase=False)

print("X dtype:", X.dtype, " / sample:", X.iloc[:3].tolist())
print("y dtype:", y.dtype, " / counts:\n", y.value_counts())
print((y.astype(str) + " 인식").iloc[:5].tolist())


# --- 모델 정의 (이 부분만 추가) ---

# 1. 로지스틱 회귀 모델 (기본 모델)
log_reg_model = LogisticRegression(solver='liblinear', random_state=42)
print("\n로지스틱 회귀 모델 정의 완료.")

# 2. 선형 SVM 모델 (확률 보정이 필요)
svc_model = CalibratedClassifierCV(LinearSVC(random_state=42))
print("선형 SVM 모델 정의 완료.")

# 3. 랜덤 포레스트 모델 (앙상블 모델)
rf_model = RandomForestClassifier(random_state=42)
print("랜덤 포레스트 모델 정의 완료.")

# 4. LightGBM 모델 (부스팅 계열, 설치 후 사용)
lgbm_model = lgb.LGBMClassifier(random_state=42)
print("LightGBM 모델 정의 완료.")

# 5. XGBoost 모델 (부스팅 계열, 설치 후 사용)
xgb_model = xgb.XGBClassifier(random_state=42)
print("XGBoost 모델 정의 완료.")
