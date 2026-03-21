# KNHANES 연령×성별 교차 층화 복합건강등급 산정 파이프라인

**논문:** 양문일 (산업정책연구원). "설명 가능한 인공지능 기반 연령×성별 교차 층화 복합건강등급 산정 방법론 — 국민건강영양조사 2020–2024를 중심으로." *한국콘텐츠학회논문지* (투고 중)

---

## 개요

국민건강영양조사(KNHANES) 2020–2024 데이터를 활용하여 연령군(중장년 40–59세 / 고령 60세 이상)과 성별을 교차한 **4개 층화 집단별** 당뇨·고혈압 예측 모형과 K-means 기반 복합건강등급을 산정한 연구의 전체 분석 코드입니다.

주요 분석 흐름은 다음과 같습니다.

1. 데이터 전처리 및 연령×성별 층화 분할
2. RF / LightGBM / XGBoost / MLP / TabNet 알고리즘 비교 (Optuna 하이퍼파라미터 최적화)
3. K-means 복합건강등급 산정 (엘보우 곡선 → K=3)
4. SHAP 변수 중요도 분석 (TreeExplainer / KernelExplainer)

---

## 레퍼지토리 구조

```
knhanes-stratified-health-grade/
├── knhanes_stratified_health_grade_pipeline.ipynb   # 전체 분석 코드 (단일 노트북)
├── HN20_ALL.sav   # KNHANES 2020 (직접 준비 필요)
├── HN21_ALL.sav   # KNHANES 2021 (직접 준비 필요)
├── HN22_ALL.sav   # KNHANES 2022 (직접 준비 필요)
├── HN23_ALL.sav   # KNHANES 2023 (직접 준비 필요)
├── HN24_ALL.sav   # KNHANES 2024 (직접 준비 필요)
├── requirements.txt
└── README.md
```

> KNHANES 원시 데이터는 저작권 및 이용 약관상 본 레퍼지토리에 포함하지 않습니다.
> 데이터 파일은 노트북과 **같은 폴더**에 위치시켜야 합니다.
> 데이터 취득 방법은 아래 **데이터 준비** 섹션을 참고하세요.

---

## 환경 설정

Python 3.9 이상 권장

```bash
git clone https://github.com/miyang0628/knhanes-stratified-health-grade.git
cd knhanes-stratified-health-grade
pip install -r requirements.txt
```

**주요 의존 패키지**

| 패키지 | 용도 |
|--------|------|
| scikit-learn | RF, MLP, K-means, 전처리 |
| lightgbm | LightGBM 모형 |
| xgboost | XGBoost 모형 |
| pytorch-tabnet | TabNet 모형 |
| shap | SHAP 변수 중요도 분석 |
| optuna | 베이지안 하이퍼파라미터 최적화 |
| pandas / numpy | 데이터 처리 |
| matplotlib / seaborn | 시각화 |

---

## 데이터 준비

1. 질병관리청 국민건강영양조사 홈페이지(https://knhanes.kdca.go.kr) 접속
2. 원시자료 신청 후 2020–2024년 5개년 데이터 다운로드
3. 다운로드한 파일을 노트북(`knhanes_stratified_health_grade_pipeline.ipynb`)과 **같은 폴더**에 위치

---

## 실행 방법

```bash
jupyter notebook knhanes_stratified_health_grade_pipeline.ipynb
```

노트북은 아래 순서로 구성되어 있습니다.

| 섹션 | 내용 |
|------|------|
| 1. 데이터 로드 및 전처리 | 5개년 병합, 결측치 처리, 변수 코딩 |
| 2. 층화 분할 | 연령×성별 4개 집단 구성 |
| 3. 모형 학습 및 비교 | 5개 알고리즘 × 4개 집단 × 2개 질환 |
| 4. K-means 복합건강등급 | 엘보우 곡선, 군집 산정, 유병률 검증 |
| 5. SHAP 분석 | 전역·국소 해석, 집단 간 비교 |

---

## 주요 결과 요약

| 집단 | 최적 모형 (당뇨) | CV-AUC | 최적 모형 (고혈압) | CV-AUC |
|------|----------------|--------|------------------|--------|
| 중장년 남성 | XGBoost | 0.753 ± 0.049 | XGBoost | 0.764 ± 0.021 |
| 중장년 여성 | LightGBM | 0.834 ± 0.023 | LightGBM | 0.744 ± 0.030 |
| 고령 남성 | RF | 0.640 ± 0.031 | LightGBM | 0.693 ± 0.026 |
| 고령 여성 | XGBoost | 0.699 ± 0.023 | RF | 0.731 ± 0.015 |

복합건강등급(K=3) 판별 타당성: 전 집단에서 1→3등급 방향 유병률 단조 증가 확인

---
