import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import numpy as np

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