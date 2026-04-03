# AI Governance Framework for Chronic Disease Risk Grading
**공정성·견고성·투명성 기반 AI 거버넌스 프레임워크: 건강검진 데이터를 활용한 집단별 만성질환 위험 등급화**

**An AI Governance Framework Based on Fairness, Robustness, and Transparency:**  
**Group-Stratified Chronic Disease Risk Grading Using National Health Examination Data**

---

## 📌 개요

KNHANES(국민건강영양조사) 2020–2024년 5개년 데이터를 활용하여 **성별 × 연령군 교차 6개 층화 집단**별 당뇨·고혈압 위험 예측 모형을 구축하고, K-means 기반 복합건강등급 산정 방법론을 제안합니다. 나아가 **공정성(G1)·견고성(G2)·투명성(G3)·설명책임(G4)** 4개 원칙으로 구성된 AI 거버넌스 프레임워크를 정량적으로 평가합니다.

> **논문:** [저자명], "[논문 제목]", [저널명], [연도] *(링크 추가 예정)*

---

## 🏗️ 파일 구조

```
knhanes-stratified-health-grade/
│
├── part1_modeling_pipeline_v2.1.py     # 데이터 전처리 → 모델 학습/평가 → K-means 등급 → SHAP
├── part2_governance_evaluation_v2.1.py # 거버넌스 정량 평가(G1–G4) → 시각화(그림 2–12)
│
├── data/
│   └── README_data.md                  # KNHANES 원시 데이터 접근 안내 (데이터 미포함)
│
├── outputs/                            # 실행 후 자동 생성
│   ├── knhanes_preprocessed.csv
│   ├── 표1_집단별기술통계.csv
│   ├── 표2_알고리즘성능비교_HO.csv
│   ├── 표3_알고리즘성능비교_CV.csv
│   ├── 표4_복합건강등급요약.csv
│   ├── 표5_거버넌스원칙정의.csv
│   ├── 표6_거버넌스정량평가종합.csv
│   ├── 표7_이해관계자역할정의.csv
│   ├── 표8_거버넌스운영사이클.csv
│   ├── 연구한계점_메타.csv
│   ├── 복합건강등급_전체.csv
│   ├── 거버넌스_지표_종합_v2.1.csv
│   ├── 그림2_집단별유병률.png
│   ├── 그림3_AUC히트맵.png
│   ├── 그림4_엘보우곡선.png
│   ├── 그림5_복합건강등급분포.png
│   ├── 그림6_SHAP_Summary_당뇨.png
│   ├── 그림7_SHAP_Summary_고혈압.png
│   ├── 그림8_SHAP집단비교.png
│   ├── 그림9_공정성지표_v2.1.png
│   ├── 그림10_거버넌스레이더_v2.1.png
│   ├── 그림11_Perturbation견고성.png
│   └── 그림12_거버넌스운영사이클.png
│
└── requirements.txt
```

> **Note:** Part 1 실행 후 직렬화 파일(`df_final.parquet`, `*.pkl`)이 생성되며, Part 2는 이를 불러와 실행합니다.

---

## 🔬 연구 방법 요약

### 데이터
| 항목 | 내용 |
|------|------|
| 출처 | KNHANES 2020–2024 원시 데이터 (`hn20_all.sas7bdat` ~ `hn24_all.sas7bdat`) |
| 대상 | 19세 이상 성인 11,467명 |
| 타겟 변수 | 당뇨 (`HE_DM_HbA1c`: 코드 1→0, 3→1), 고혈압 (`HE_HP`: 코드 1→0, 4→1) |
| 예측 변수 | 수치형 12개 + 범주형 19개 = **총 31개** |

### 층화 집단 (6개)
| 집단 | 연령 | 성별 | 표본 수 |
|------|------|------|---------|
| 청년_남성 | 19–39세 | 남 | 1,347 |
| 청년_여성 | 19–39세 | 여 | 2,647 |
| 중장년_남성 | 40–59세 | 남 | 1,273 |
| 중장년_여성 | 40–59세 | 여 | 2,624 |
| 고령_남성 | 60세↑ | 남 | 1,569 |
| 고령_여성 | 60세↑ | 여 | 2,007 |

> ⚠️ 청년 집단은 당뇨·고혈압 유병자 수가 적어(당뇨: 남성 37명, 여성 36명) 소표본 경고가 표시됩니다.

### 알고리즘 및 최적화
| 항목 | 내용 |
|------|------|
| 비교 알고리즘 | LR (Baseline), RF, LightGBM, XGBoost, MLP |
| 최적 모델 선정 | LR 제외 4개 알고리즘 중 AUC 최고값 기준 |
| 하이퍼파라미터 최적화 | Optuna 베이즈 최적화 (`n_trials=20`, 위험 판정 시 50) |
| 클래스 불균형 처리 | 표본 가중치 (sample weight) — SMOTE 미사용 |
| 검증 방식 | Hold-out (7:3 층화분할) + 5-겹 교차검증 병행 |

---

## 📊 주요 결과

### 모델 성능 (Hold-out AUC)
| 집단 | 질환 | 최적 알고리즘 | AUC | LR Baseline | 향상폭 |
|------|------|--------------|-----|-------------|--------|
| 청년_여성 | 당뇨 | RF | 0.890 | 0.611 | **+0.279** |
| 청년_여성 | 고혈압 | RF | 0.836 | 0.641 | +0.195 |
| 청년_남성 | 고혈압 | RF | 0.825 | 0.690 | +0.134 |
| 중장년_여성 | 당뇨 | LGBM | 0.833 | 0.827 | +0.006 |
| 고령_여성 | 고혈압 | RF | 0.779 | 0.727 | +0.053 |
| 고령_남성 | 당뇨 | RF | 0.638 | 0.594 | +0.045 |

### 복합건강등급 (K-means, k=3)
등급이 높아질수록 실제 유병률 단조 증가 → **판별 타당성 전 집단 확인**

| 집단 | 3등급(고위험) 당뇨 유병률 | 3등급(고위험) 고혈압 유병률 |
|------|--------------------------|--------------------------|
| 고령_남성 | **73.5%** | **94.9%** |
| 고령_여성 | 75.2% | 87.6% |
| 청년_남성 | 36.8% | 32.9% |

### AI 거버넌스 정량 평가 (G1–G4)
| 원칙 | 지표 | 당뇨 | 고혈압 | 판정 |
|------|------|------|--------|------|
| **G1 공정성** | Δ-AUC (기준 < 0.10) | 0.252 | 0.150 | ❌ / ⚠️ |
| **G1 공정성** | Equalized Odds Gap (기준 < 0.15) | 0.921 | 1.251 | ❌ 위험 |
| **G1 공정성** | ECE Gap (기준 < 0.05) | 0.117 | 0.266 | ❌ 위험 |
| **G2 견고성** | CV Gap 평균 (기준 < 0.03) | 0.028 | 0.039 | ✅ / ⚠️ |
| **G2 견고성** | Perturbation Drop (기준 < 0.03) | 0.011 | 0.019 | ✅ 양호 |
| **G3 투명성** | SHAP HHI (기준 < 0.15) | 0.126 | 0.092 | ✅ 양호 |
| **G4 설명책임** | Model Card Score (기준 ≥ 0.80) | 0.80 (8/10) | 0.80 (8/10) | ✅ 양호 |
| **종합** | 가중 종합 점수 (기준 ≥ 0.75) | **0.65** | **0.70** | ⚠️ 주의 |

> G1 공정성 위험 판정의 주요 원인: 청년 소표본 집단의 TPR 극단값 (청년_여성 고혈압 TPR=0.00) 및 집단 간 유병률 이질성

---

## ⚙️ 설치 및 실행

### 요구 환경
```
Python >= 3.9
```

### 패키지 설치
```bash
git clone https://github.com/miyang0628/knhanes-stratified-health-grade.git
cd knhanes-stratified-health-grade
pip install -r requirements.txt
```

### 주요 패키지
```
scikit-learn
lightgbm
xgboost
optuna
shap
pyreadstat
pandas
numpy
joblib
matplotlib
```

### 실행 순서

**Step 1. KNHANES 원시 데이터 준비**  
아래 안내에 따라 SAS 파일을 프로젝트 루트에 위치시킵니다.
```
hn20_all.sas7bdat  # KNHANES 2020
hn21_all.sas7bdat  # KNHANES 2021
hn22_all.sas7bdat  # KNHANES 2022
hn23_all.sas7bdat  # KNHANES 2023
hn24_all.sas7bdat  # KNHANES 2024
```

**Step 2. Part 1 실행** (데이터 전처리 → 모델 학습 → K-means 등급 → SHAP)
```bash
python part1_modeling_pipeline_v2.1.py
```

**Step 3. Part 2 실행** (거버넌스 G1–G4 평가 → 그림 2–12 생성)
```bash
python part2_governance_evaluation_v2.1.py
```

> Part 1 완료 시 `df_final.parquet` 및 `*.pkl` 직렬화 파일이 생성됩니다.  
> Part 2는 이 파일들을 자동으로 불러옵니다.

---

## 📁 주요 출력물

### 논문용 표 (CSV)
| 파일 | 내용 |
|------|------|
| `표1_집단별기술통계.csv` | 집단별 표본 수, 유병률, 소표본 경고 |
| `표2_알고리즘성능비교_HO.csv` | Hold-out AUC (LR Baseline 포함) |
| `표3_알고리즘성능비교_CV.csv` | 5-겹 CV AUC (mean ± std) + CV Gap |
| `표4_복합건강등급요약.csv` | K-means 등급별 분포·유병률·정책 함의 |
| `표5_거버넌스원칙정의.csv` | G1–G4 임계값·규제근거·도메인적합성 주석 |
| `표6_거버넌스정량평가종합.csv` | G1–G4 수치·판정·규제근거 통합 |
| `표7_이해관계자역할정의.csv` | 역할·주체·책임·트리거·산출물 |
| `표8_거버넌스운영사이클.csv` | 연간 6단계 운영 절차 상세 |
| `연구한계점_메타.csv` | 5개 한계유형별 내용·영향·향후과제 |

### 논문용 그림 (PNG, DPI=200)
| 파일 | 내용 |
|------|------|
| `그림2_집단별유병률.png` | 6개 집단 당뇨·고혈압 유병률 |
| `그림3_AUC히트맵.png` | 알고리즘별 AUC 히트맵 (LR Baseline 포함, ★ 최적) |
| `그림4_엘보우곡선.png` | K-means k=3 선정 근거 |
| `그림5_복합건강등급분포.png` | 등급별 구성비 + 유병률 |
| `그림6_SHAP_Summary_당뇨.png` | 집단별 SHAP Beeswarm (당뇨) |
| `그림7_SHAP_Summary_고혈압.png` | 집단별 SHAP Beeswarm (고혈압) |
| `그림8_SHAP집단비교.png` | 집단 간 변수 중요도 비교 |
| `그림9_공정성지표_v2.1.png` | AUC 공정성 + Equalized Odds |
| `그림10_거버넌스레이더_v2.1.png` | G1–G4 가중 종합 레이더 차트 |
| `그림11_Perturbation견고성.png` | ±10% 노이즈 AUC Drop 히트맵 |
| `그림12_거버넌스운영사이클.png` | 연간 거버넌스 운영 사이클 플로우차트 |

---

## 📋 데이터 접근 안내

KNHANES 원시 데이터는 저작권 및 개인정보보호 정책에 따라 본 저장소에 포함되지 않습니다.  
공식 홈페이지에서 직접 신청·다운로드 후 사용하십시오.

🔗 **[질병관리청 KNHANES 공식 다운로드](https://knhanes.kdca.go.kr)**

---

## 🏛️ AI 거버넌스 프레임워크

| 원칙 | 주요 지표 | 규제 근거 | 선행연구 |
|------|----------|----------|---------|
| G1 공정성 | Δ-AUC, Equalized Odds Gap, ECE Gap | EU AI Act Art.10, 금융위 AI 가이드라인 | Chouldechova (2017), Hardt et al. (2016) |
| G2 견고성 | CV Gap, Temporal Gap, Perturbation Drop | OECD AI 원칙 §견고성·안전성 | Varma & Simon (2006), Ghorbani & Zou (2019) |
| G3 투명성 | SHAP HHI (Bootstrap 95% CI) | 개인정보보호법 제37조의2 | Lundberg & Lee (2017) |
| G4 설명책임 | Model Card Score (10개 항목) | EU AI Act Art.13, AI기본법(2026.1.21. 시행) | Mitchell et al. (2019) |

**임계값 도메인 적합성 주석:**  
G1 기준은 금융 AI 기준을 차용하였으나, 6개 성별×연령군 층화로 구조적 유병률 이질성이 존재하므로 단일집단 기준 대비 완화 적용하였습니다(Wachter et al., 2021). 의료 AI 전용 거버넌스 임계값 체계 수립은 향후 연구과제로 제안합니다.

---

## 📄 인용

본 코드를 연구에 활용하시는 경우 아래 논문을 인용해 주십시오.

```bibtex
@article{[author]2025,
  title   = {공정성·견고성·투명성 기반 AI 거버넌스 프레임워크:
             건강검진 데이터를 활용한 집단별 만성질환 위험 등급화},
  author  = {[저자명]},
  journal = {[저널명]},
  year    = {2025}
}
```

---

## 📬 문의

Issues 탭 또는 이메일([your-email@domain.com])로 문의해 주십시오.

---

## 📜 라이선스

This project is licensed under the MIT License. See `LICENSE` for details.
