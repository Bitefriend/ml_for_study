# NVIDIA에서 직접 제공하는 CUDA 기반 이미지를 사용합니다.
# 이 이미지에는 GPU를 사용하기 위한 모든 드라이버와 라이브러리가 포함되어 있습니다.
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# 필요한 시스템 패키지들을 설치합니다.
# 파이썬과 파이썬 패키지 관리자인 pip을 설치합니다.
RUN apt-get update && apt-get install -y python3 python3-pip

# 필요한 파이썬 패키지들을 모두 설치합니다.
# XGBoost, LightGBM, 그리고 GPU 모니터링을 위한 pynvml 등이 포함됩니다.
RUN pip install -U numpy pandas scikit-learn packaging xgboost lightgbm pynvml