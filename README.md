# 악성 URL 분류 AI 경진대회 (DACON)

> **대회 링크:** https://dacon.io/competitions/official/236451/overview/description  
> 본 README는 위 대회 페이지(캡처 기준) 내용을 바탕으로 작성되었습니다. 세부 규정은 **항상 대회 페이지 공지**를 최종 기준으로 하며, 변경 시 본 문서도 업데이트합니다.

---

## Overview

- **태스크**: 주어진 URL의 **악성 여부(0=정상, 1=악성)**를 예측하는 이진 분류
- **도메인**: 사이버 보안 / NLP (문자열 기반 URL 분석)
- **활용 분야**: 피싱/멀웨어 유포 링크 탐지, 이메일/메신저 보안 필터링, 보안 게이트웨이 선제 차단 등
- **데이터**: `train.csv`(ID, URL, label), `test.csv`(ID, URL), 제출용 `sample_submission.csv`
- **평가 지표**: **ROC-AUC**  
  - Public: 테스트 데이터의 약 30%  
  - Private: 테스트 데이터의 약 70%

> **기간(배너 기준)**: 2025-02-03 ~ 2025-03-31 09:59  
> **상금(배너 표기)**: 데이스쿨 프로 구독권  
> (정확한 세부 조건은 대회 페이지 확인 필수)

---

## Team

| [조재형](https://github.com/Bitefriend) | [김영](https://github.com/kimyoung9689) |
|:--:|:--:|
| 팀장 | 팀원 |

---

## 0. Getting Started

- 본 프로젝트는 **DACON 악성 URL 분류 AI 경진대회** 참가를 위한 저장소입니다.
- 아래 항목 중 **사용자가 제공하지 않은 정보는 모두 예시**로만 표기하고, 실제 내용은 추후 확정 시 업데이트합니다.

### Environment *(예시 — 추후 작성 예정)*
- OS: 예시) Linux / macOS
- Python: 예시) 3.11+
- Package Manager: 예시) Poetry 또는 pip
- Frameworks: 예시) PyTorch / scikit-learn (필요 시 Hugging Face)
- Experiment Tracking: 예시) Weights & Biases

### Requirements *(예시 — 추후 작성 예정)*
- 필수: pandas, numpy, scikit-learn, tqdm, python-dotenv
- 선택: lightgbm/xgboost, torch, pytorch-lightning, regex, tokenizers/sentencepiece, polars/dask, wandb, matplotlib, seaborn

### 설치 *(예시 — 추후 작성 예정)*
```bash
# 예시) Poetry
poetry install
# 또는 pip
pip install -r requirements.txt
```

### 환경 변수 *(예시 — 추후 작성 예정)*
```bash
cp .env.template .env
vi .env
```
- 예시) `PYTHONPATH`, `WANDB_*` 등

### 데이터 다운로드
- 대회 페이지에서 **train.csv / test.csv / sample_submission.csv**를 수동 다운로드하여 배치합니다. (링크: 대회 페이지 참조)

## 1. Competition Info

### Dataset

- **train.csv**: 약 **6,995,056 rows**  
  - `ID` (샘플 고유 ID)  
  - `URL` (문자열)  
  - `label` (0=정상, 1=악성)
- **test.csv**: 약 **1,747,689 rows**  
  - `ID`, `URL`
- **sample_submission.csv**:  
  - `ID`, `probability` (= 악성일 확률, [0,1])

> 위 수치는 페이지/스크린샷 기준이며, 공식 수치는 대회 페이지를 최종 확인하세요.

### 제출 규격

- 컬럼: `ID, probability`  
- 파일 형식: `csv`  
- 스코어 산출: **ROC-AUC** (Private가 최종 순위)

### 규정 요약(핵심만 발췌)

- **참여**: 개인 또는 팀 참여 가능(동일인의 중복 등록 불가).  
- **제출 횟수**: 하루 최대 **5회**.  
- **외부 데이터**: **불가**(대회 제공 데이터 외 사용 금지).  
- **사전학습 모델**: **공식적으로 공개**되어 있고 **법적 제약 없는 가중치**는 사용 가능.  
- **코드 & PPT**: 최종 검증 시 **Private 상위 10팀** 대상, 재현 가능한 **코드 + 솔루션 설명 PPT** 제출/검증.  
- 모든 세부 규정·예외 사항은 **대회 페이지 공지**를 반드시 확인하세요.

---

## 2. 프로젝트 구조 *(예시 — 추후 작성 예정)*

```
project/
├── data/
│   ├── raw/                     # train.csv, test.csv, sample_submission.csv
│   └── processed/               # 전처리 산출물 (예: parquet)
├── docs/
│   ├── eda.md                   # EDA 결과 (추후 작성 예정)
│   ├── modeling.md              # 실험/모델 정리 (추후 작성 예정)
│   └── presentation.pdf         # 발표자료 (추후 작성 예정)
├── notebooks/
│   └── exploration.ipynb        # 탐색 노트북 (추후 작성 예정)
├── src/
│   ├── config/                  # 설정 (추후 작성 예정)
│   ├── data/                    # 로더/전처리 (추후 작성 예정)
│   ├── features/                # 피처엔지니어링 (추후 작성 예정)
│   ├── models/                  # 모델 (추후 작성 예정)
│   ├── training/                # 학습 루프 (추후 작성 예정)
│   ├── inference/               # 추론/제출 (추후 작성 예정)
│   └── scripts/                 # 실행 스크립트 (추후 작성 예정)
└── README.md
```

## 3. 방법론 로드맵 *(초안 예시 — 추후 작성 예정)*
- 베이스라인: 예시) char-level TF-IDF/Hashing + LogisticRegression/LinearSVC
- 대안: 예시) LightGBM/XGBoost, 경량 CNN/Transformer
- 평가/튜닝: 예시) Stratified K-Fold, 임계값 조정, 앙상블(soft-voting)

> 실제 확정 모델/전략은 실험 후 `docs/modeling.md`에 기록합니다. (추후 작성 예정)


대용량(약 700만 행) 특성상 **경량·확장성** 중심으로 설계합니다.

### A) 고전 ML 베이스라인 (권장 시작점)

- **전처리**: URL 정규화(스킴/포트 제거, Punycode/Percent-decoding 비율, 길이, 숫자/특수문자 비율 등)
- **텍스트 피처**: `HashingVectorizer` 또는 `TfidfVectorizer`(char n-gram 3~5 권장)
- **모델**: `LogisticRegression`(saga), `LinearSVC`, `LightGBM/XGBoost`
- **장점**: 훈련/추론 빠름, 메모리 관리 용이, 대용량에 적합

### B) 딥러닝(Char-level)

- **토크나이즈**: 문자 단위 인덱싱
- **모델**: 1D-CNN / 작은 Transformer / GRU
- **전략**: 입력 길이 클리핑, `AMP`, `gradient_accumulation`

### C) 앙상블

- 서로 상관이 낮은 모델들의 **soft-voting** 또는 **stacking**

> 각 방법의 상세 실험 기록은 `docs/modeling.md`에 누적합니다. *(추후 작성 예정)*

---

## 4. 스크립트 사용법 *(예시 — 추후 작성 예정)*

> 아래는 **형태 예시**입니다. 실제 파일명/옵션은 구현 후 확정합니다.

### 4.1 전처리 & 피처링 *(예시)*
```bash
python src/scripts/make_features.py   --input data/raw/train.csv   --output data/processed/train_features.parquet   --vectorizer hashing   --n-gram 3 5
```

### 4.2 학습 *(예시)*
```bash
python src/scripts/train.py   --features data/processed/train_features.parquet   --model lr   --seed 4321
```

### 4.3 추론 & 제출파일 생성 *(예시)*
```bash
python src/scripts/predict.py   --features data/processed/test_features.parquet   --checkpoint outputs/lr_fold0.bin   --out submissions/submission_v1.csv
```

## 5. EDA 체크리스트 *(초안 예시 — 추후 작성 예정)*
- 샘플 길이/문자 분포, 레이블 비율
- URL 구조 통계(호스트/경로/쿼리, IP/포트, 특수문자 비율)
- 인코딩(Punycode/Percent-encoding) 비율
- 중복/유사도 처리 정책
- 스플릿 전략: Stratified K-Fold 또는 시드 고정 홀드아웃

## 6. 제출 & 리더보드

1. `submissions/submission_*.csv` 생성  
2. DACON 페이지 업로드 → Public/Private 점수 확인  
3. 하루 제출 제한 **5회** 유의

> 최종 순위는 **Private ROC-AUC**로 결정됩니다.

---

## 7. 재현성 & 규정 준수 체크리스트

- [ ] 코드로 **완전 재현** 가능 (seed, 환경, 버전 명시)
- [ ] 외부 데이터 **미사용** 확인
- [ ] 사전학습 가중치 라이선스 확인(허용 범위 내)
- [ ] Private 상위권 진입 시 **코드 + PPT** 제출 준비  
      - 실행 가이드, 의존성 파일(`requirements.txt`/`pyproject.toml`),
        환경 버전 정보, 학습/추론 스크립트 포함
- [ ] 제출 파일 포맷 검증(`ID,probability`)

---

## 8. 성능 향상 아이디어 보관함 *(아이디어 예시 — 추후 작성 예정)*
- URL 정규화 규칙 엔지니어링(축약링크 해제, 서브도메인 깊이 등)
- char n-gram/벡터화 비교(Hashing vs TF-IDF)
- Class weight/threshold 최적화
- 앙상블/스태킹

## 9. 로그 & 문서

- `docs/eda.md` — EDA 결과 *(추후 작성 예정)*
- `docs/modeling.md` — 모델/하이퍼파라미터/결과 *(추후 작성 예정)*
- `docs/presentation.pdf` — 최종 발표자료 *(추후 작성 예정)*

---

## 10. 변경 이력(Changelog)

- 2025-08-11: 초기 README(NLP/보안 대회 버전) 생성

---

## 11. 라이선스

- 이 저장소의 코드 라이선스는 `LICENSE` 파일을 따릅니다. (추후 명시 예정)

---

### 참고

- 모든 규정과 수치는 **대회 페이지** 공지를 최종 기준으로 합니다.  
- 본 README는 참여/개발 편의를 위한 가이드이며, 오해 소지가 있는 표현은 발견 즉시 수정합니다.
