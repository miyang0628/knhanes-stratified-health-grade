# ============================================================
# [Part 2] 거버넌스 정량 평가 + 시각화 — 업그레이드 v2.1
# Part 1 실행 후 이어서 실행 (동일 세션, 동일 노트북)
#
# 추가/변경 사항 (v2.1):
#   - 표5: 거버넌스 원칙 정의 및 임계값 (표5_거버넌스원칙정의.csv)
#   - 표6: G1–G4 정량 평가 종합 (표6_거버넌스정량평가종합.csv)
#   - 표7: 이해관계자 역할 정의 (표7_이해관계자역할정의.csv)
#   - 표8: 거버넌스 운영 사이클 단계 (표8_거버넌스운영사이클.csv)
#   - 그림12: 거버넌스 운영 사이클 플로우차트 (신규)
#   - 임계값 도메인 적합성 근거 주석 보강
#
# 필요 변수 (Part 1에서 생성):
#   df_final, all_results, best_models, best_algo_name,
#   cv_df, grade_results, shap_results,
#   X_FEATURES, DISEASES, AGEGROUP_CONFIG, GRP_NAMES, GRP_LABEL,
#   COLORS, ALGOS, SEED, DPI, GITHUB_URL
# ============================================================

import matplotlib
matplotlib.use('Agg')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

plt.rcParams['font.family']        = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus']  = False

# ── Part 1 결과물 불러오기 ──
df_final       = pd.read_parquet('df_final.parquet')
all_results    = joblib.load('all_results.pkl')
best_models    = joblib.load('best_models.pkl')
best_algo_name = joblib.load('best_algo_name.pkl')
shap_results   = joblib.load('shap_results.pkl')
grade_results  = joblib.load('grade_results.pkl')
cv_df          = joblib.load('cv_df.pkl')

# ── 상수 재정의 ──
SEED = 42
DPI  = 200
GITHUB_URL = "https://github.com/miyang0628/knhanes-stratified-health-grade"

AGEGROUP_CONFIG = {
    "청년_남성":   {"age_min": 19, "age_max": 39, "sex_code": 1.0, "age_group": 0.0},
    "청년_여성":   {"age_min": 19, "age_max": 39, "sex_code": 2.0, "age_group": 0.0},
    "중장년_남성": {"age_min": 40, "age_max": 59, "sex_code": 1.0, "age_group": 1.0},
    "중장년_여성": {"age_min": 40, "age_max": 59, "sex_code": 2.0, "age_group": 1.0},
    "고령_남성":   {"age_min": 60, "age_max": 99, "sex_code": 1.0, "age_group": 2.0},
    "고령_여성":   {"age_min": 60, "age_max": 99, "sex_code": 2.0, "age_group": 2.0},
}
GRP_NAMES = list(AGEGROUP_CONFIG.keys())
GRP_LABEL = ["청년 남성", "청년 여성", "중장년 남성",
             "중장년 여성", "고령 남성", "고령 여성"]
COLORS    = ['#1565C0', '#90CAF9', '#E65100',
             '#FFCC80', '#2E7D32', '#A5D6A7']
DISEASES  = {'Diabetes': '당뇨', 'Hypertension': '고혈압'}
ALGOS     = ['LR', 'RF', 'LGBM', 'XGB', 'MLP']
X_FEATURES = [
    'BMI', 'WaistCirc', 'Weight',
    'Energy_kcal', 'Carb_g', 'Sugar_g', 'Sodium_mg',
    'Fat_g', 'SaturatedFat_g', 'Fiber_g', 'Potassium_mg', 'Protein_g',
    'ObesityStatus', 'WeightChangeStatus', 'WeightLossAmount', 'WeightGainAmount',
    'DrinkingFrequency', 'DrinkingAmount', 'SmokingStatus',
    'VigorousActivity_Work', 'VigorousActivity_Leisure', 'ModerateActivity_Work',
    'WalkingActivity', 'AerobicActivityRate', 'BreakfastFrequency',
    'StressLevel', 'StressAwarenessRate', 'PersonalIncomeQuartile',
    'HouseholdIncomeQuartile', 'EducationLevel', 'HealthScreeningStatus',
]

# ── 거버넌스 임계값 ──
# ※ 임계값 도메인 적합성 주석:
#    G1 기준은 금융 AI(Chouldechova 2017, Hardt 2016)에서 차용하였으나,
#    본 연구는 6개 성별×연령군 층화로 구조적 유병률 이질성이 존재하므로
#    단일집단 기준 대비 완화 적용(Δ-AUC 0.05→0.10, EqOdds 0.10→0.15).
#    의료 AI 전용 거버넌스 임계값 체계는 향후 연구과제로 제안.
THRESHOLDS = {
    'G1_delta_auc':    {'good': 0.10, 'warn': 0.20,
        'cite': 'Chouldechova(2017); 6집단 층화 시 구조적 격차 감안 완화 적용'},
    'G1_eq_odds':      {'good': 0.15, 'warn': 0.30,
        'cite': 'Hardt et al.(2016); 6집단 유병률 이질성 반영 조정'},
    'G1_calib_ece':    {'good': 0.05, 'warn': 0.10,
        'cite': 'Naeini et al.(2015) ECE 기준, 의료 AI 권고'},
    'G2_cv_gap':       {'good': 0.03, 'warn': 0.05,
        'cite': 'Varma & Simon(2006) nested CV 안정성 기준'},
    'G2_temporal_gap': {'good': 0.05, 'warn': 0.10,
        'cite': 'Nestor et al.(2019) temporal shift 허용 범위'},
    'G2_perturb_drop': {'good': 0.03, 'warn': 0.05,
        'cite': 'Ghorbani & Zou(2019) data robustness 기준'},
    'G3_hhi':          {'good': 0.15, 'warn': 0.25,
        'cite': 'DOJ/FTC HHI 기준 차용, Lundberg et al.(2020) 권고'},
    'G4_model_card':   {'good': 0.80, 'warn': 0.60,
        'cite': 'Mitchell et al.(2019) Model Cards for Model Reporting'},
}

def judge(val, key, low_best=True):
    t = THRESHOLDS[key]
    if low_best:
        if val <= t['good']: return '양호 O'
        if val <= t['warn']: return '주의 -'
        return '위험 X'
    else:
        if val >= t['good']: return '양호 O'
        if val >= t['warn']: return '주의 -'
        return '위험 X'

def norm_score(val, good, bad, floor=0.05):
    if val <= good: return 1.0
    if val >= bad:  return floor
    return floor + (1.0 - floor) * (1.0 - (val - good) / (bad - good))

MODEL_CARD_CHECKLIST = {
    '모델 목적 및 의도된 용도 명시':        True,
    '학습 데이터 출처 및 기간 명시':         True,
    '평가 지표 및 결과 보고':               True,
    '성능 한계 및 실패 사례 명시':           True,
    '집단별 성능 분해(disaggregation)':     True,
    'SHAP 개별 설명(Waterfall) 가용':       True,
    '코드 공개(GitHub)':                    True,
    '인간 감독 절차(human-in-the-loop)':    False,   # 향후 과제
    '모델 버전 관리 및 갱신 계획':           True,
    '개인정보 영향평가(PIA) 수행':           False,   # 향후 과제
}


# %% =========================================================
# § 표5  거버넌스 원칙 정의 및 임계값
# ============================================================

table5_rows = [
    {
        '원칙': 'G1 공정성',
        '지표': 'Δ-AUC (집단간 AUC 격차)',
        '양호기준': 'Δ < 0.10',
        '주의기준': 'Δ < 0.20',
        '완화적용근거': '6집단 층화로 구조적 유병률 이질성 존재 — 단일집단 기준(0.05) 완화',
        '선행연구': 'Chouldechova (2017)',
        '규제근거': 'EU AI Act Art.10 §데이터 거버넌스',
        '도메인적합성_주석': '금융 AI 기준 차용; 의료 전용 기준 향후 연구 필요',
    },
    {
        '원칙': 'G1 공정성',
        '지표': 'Equalized Odds Gap (TPR+FPR 격차합)',
        '양호기준': 'EO < 0.15',
        '주의기준': 'EO < 0.30',
        '완화적용근거': '6집단 유병률 이질성 반영 — 단일집단 기준(0.10) 완화',
        '선행연구': 'Hardt et al. (2016)',
        '규제근거': '금융위 AI 가이드라인 §공정성',
        '도메인적합성_주석': '금융 AI 기준 차용; 의료 전용 기준 향후 연구 필요',
    },
    {
        '원칙': 'G1 공정성',
        '지표': 'ECE Gap (보정오차 집단간 격차)',
        '양호기준': 'ECE < 0.05',
        '주의기준': 'ECE < 0.10',
        '완화적용근거': '의료 AI 표준 ECE 기준 직접 적용',
        '선행연구': 'Naeini et al. (2015)',
        '규제근거': '의료기기 소프트웨어 가이드라인',
        '도메인적합성_주석': '의료 AI 권고 기준 — 도메인 적합성 높음',
    },
    {
        '원칙': 'G2 견고성',
        '지표': 'CV vs HO AUC Gap',
        '양호기준': '|Gap| < 0.03',
        '주의기준': '|Gap| < 0.05',
        '완화적용근거': '표준 nested CV 안정성 기준 직접 적용',
        '선행연구': 'Varma & Simon (2006)',
        '규제근거': 'OECD AI 원칙 §견고성·안전성',
        '도메인적합성_주석': '통계학적 범용 기준 — 도메인 적합성 높음',
    },
    {
        '원칙': 'G2 견고성',
        '지표': 'Temporal AUC Gap (2024년 검증)',
        '양호기준': '|Gap| < 0.05',
        '주의기준': '|Gap| < 0.10',
        '완화적용근거': '횡단면 데이터 구조상 단일연도 분할 — 종단 대비 제한적',
        '선행연구': 'Nestor et al. (2019)',
        '규제근거': 'OECD AI 원칙 §견고성·안전성',
        '도메인적합성_주석': '횡단면 한계로 단일 연도 분할; 코호트 연계 시 개선 가능',
    },
    {
        '원칙': 'G2 견고성',
        '지표': 'Feature Perturbation AUC Drop (±10% 노이즈)',
        '양호기준': 'Drop < 0.03',
        '주의기준': 'Drop < 0.05',
        '완화적용근거': '데이터 내성 범용 기준 직접 적용',
        '선행연구': 'Ghorbani & Zou (2019)',
        '규제근거': 'OECD AI 원칙 §견고성·안전성',
        '도메인적합성_주석': '범용 robustness 기준 — 도메인 적합성 높음',
    },
    {
        '원칙': 'G3 투명성',
        '지표': 'SHAP HHI (변수 중요도 집중도, Bootstrap 95% CI)',
        '양호기준': 'HHI < 0.15',
        '주의기준': 'HHI < 0.25',
        '완화적용근거': 'HHI 낮을수록 다수 변수 분산 설명 — 투명성 높음',
        '선행연구': 'Lundberg et al. (2020)',
        '규제근거': '개인정보보호법 제37조의2 §설명요구권',
        '도메인적합성_주석': 'DOJ/FTC 경쟁집중도 기준 차용; XAI 설명 분산도로 재해석',
    },
    {
        '원칙': 'G4 설명책임',
        '지표': 'Model Card Score (10개 항목 체크리스트)',
        '양호기준': 'Score ≥ 0.80',
        '주의기준': 'Score ≥ 0.60',
        '완화적용근거': '10개 항목 중 8개 충족 (미충족: human-in-the-loop, PIA)',
        '선행연구': 'Mitchell et al. (2019)',
        '규제근거': 'EU AI Act Art.13 §투명성; OECD AI 원칙',
        '도메인적합성_주석': '국제 표준 Model Card 기준 — 도메인 적합성 높음',
    },
]

table5 = pd.DataFrame(table5_rows)
table5.to_csv('표5_거버넌스원칙정의.csv', index=False, encoding='utf-8-sig')
print(">>> 저장: 표5_거버넌스원칙정의.csv")


# %% =========================================================
# § G1  공정성: Δ-AUC + Equalized Odds + ECE
# ============================================================

def compute_equalized_odds(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return tpr, fpr

def compute_ece(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i+1])
        if mask.sum() == 0:
            continue
        bin_acc  = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return ece / len(y_true)

print("\n" + "="*55)
print("[G1] 공정성 지표: Δ-AUC + Equalized Odds Gap + ECE Gap")
print("="*55)

fair_records = []
for grp, cfg in AGEGROUP_CONFIG.items():
    df_g = df_final[
        (df_final['AgeGroup'] == cfg['age_group']) &
        (df_final['Sex']      == cfg['sex_code'])
    ].copy()
    for dis_col, dis_name in DISEASES.items():
        mdl  = best_models.get(grp, {}).get(dis_col)
        algo = best_algo_name.get(grp, {}).get(dis_col)
        if mdl is None:
            continue
        X = df_g[X_FEATURES]
        y = df_g[dis_col].astype(int)
        _, X_val, _, y_val = train_test_split(
            X, y, test_size=0.3, random_state=SEED, stratify=y)
        prob = mdl.predict_proba(X_val)[:, 1]
        auc  = roc_auc_score(y_val, prob)
        tpr, fpr = compute_equalized_odds(y_val.values, prob)
        ece = compute_ece(y_val.values, prob)
        fair_records.append({
            '집단': grp, '질환': dis_name, '알고리즘': algo,
            'AUC': round(auc, 4), 'TPR': round(tpr, 4),
            'FPR': round(fpr, 4), 'ECE': round(ece, 4),
            'N': len(y_val),
        })

df_fair = pd.DataFrame(fair_records)

fairness_summary_rows = []
for dis_name in ['당뇨', '고혈압']:
    sub = df_fair[df_fair['질환'] == dis_name]
    if len(sub) < 2:
        continue
    delta_auc = round(sub['AUC'].max() - sub['AUC'].min(), 4)
    delta_tpr = round(sub['TPR'].max() - sub['TPR'].min(), 4)
    delta_fpr = round(sub['FPR'].max() - sub['FPR'].min(), 4)
    eq_odds   = round(delta_tpr + delta_fpr, 4)
    delta_ece = round(sub['ECE'].max() - sub['ECE'].min(), 4)
    fairness_summary_rows.append({
        '질환': dis_name,
        'Δ_AUC': delta_auc, 'Δ_AUC_판정': judge(delta_auc, 'G1_delta_auc'),
        'EqOdds_Gap': eq_odds, 'EqOdds_판정': judge(eq_odds, 'G1_eq_odds'),
        'Δ_ECE': delta_ece, 'ECE_판정': judge(delta_ece, 'G1_calib_ece'),
        '근거': THRESHOLDS['G1_eq_odds']['cite'],
        '임계값_완화근거': '6집단 층화로 구조적 유병률 이질성 반영',
    })

fairness_summary = pd.DataFrame(fairness_summary_rows)
print(df_fair.to_string(index=False))
print("\n── 공정성 판정 요약 ──")
print(fairness_summary.to_string(index=False))


# %% =========================================================
# § G2  견고성: CV Gap + Temporal + Perturbation
# ============================================================

# ── [G2-a] 견고성: |CV_AUC − HO_AUC| ──
print("\n" + "="*55)
print("[G2-a] 견고성: |CV_AUC − HO_AUC|")
print("="*55)

# cv_df 컬럼명 확인 후 유연하게 처리
print(f"  cv_df 컬럼 확인: {cv_df.columns.tolist()}")

# 컬럼명 자동 매핑 (한글 인코딩 차이 대응)
col_grp  = [c for c in cv_df.columns if '집단' in c or '그룹' in c or 'grp' in c.lower()][0]
col_dis  = [c for c in cv_df.columns if '질환' in c or 'dis' in c.lower()][0]
col_algo = [c for c in cv_df.columns if '알고리즘' in c or 'algo' in c.lower()][0]
col_ho   = [c for c in cv_df.columns if 'HO_AUC' in c][0]
col_cv   = [c for c in cv_df.columns if 'CV_AUC' in c][0]

print(f"  매핑 → 집단:{col_grp} / 질환:{col_dis} / 알고리즘:{col_algo}")

rob_rows = []
for _, r in cv_df.iterrows():
    try:
        cv_mean = float(str(r[col_cv]).split('±')[0].strip())
    except (ValueError, TypeError):
        cv_mean = np.nan
    ho_auc = float(r[col_ho]) if r[col_ho] != '-' else np.nan
    gap = round(abs(cv_mean - ho_auc), 4) if not np.isnan(cv_mean) and not np.isnan(ho_auc) else np.nan
    rob_rows.append({
        '집단':      r[col_grp],
        '질환':      r[col_dis],
        '알고리즘':  r[col_algo],
        'HO_AUC':    ho_auc,
        'CV_AUC_mean': cv_mean,
        '|Gap|':     gap,
        '견고성_판정': judge(gap, 'G2_cv_gap') if not np.isnan(gap) else '-',
        '근거':      THRESHOLDS['G2_cv_gap']['cite'],
    })

robustness_df = pd.DataFrame(rob_rows)
print(robustness_df.to_string(index=False))

print("\n" + "="*55)
print("[G2-b] 견고성: Temporal Validation (2020–2023 → 2024)")
print("※ 주의: KNHANES 횡단면 구조상 단일연도 분할 — 종단 롤링 불가")
print("="*55)

temporal_rows = []
if 'SurveyYear' in df_final.columns:
    for grp, cfg in AGEGROUP_CONFIG.items():
        df_g = df_final[
            (df_final['AgeGroup'] == cfg['age_group']) &
            (df_final['Sex']      == cfg['sex_code'])
        ].copy()
        df_train = df_g[df_g['SurveyYear'] < 2024]
        df_test  = df_g[df_g['SurveyYear'] >= 2024]
        if len(df_test) < 30:
            continue
        for dis_col, dis_name in DISEASES.items():
            mdl  = best_models.get(grp, {}).get(dis_col)
            algo = best_algo_name.get(grp, {}).get(dis_col)
            if mdl is None:
                continue
            X_te = df_test[X_FEATURES]
            y_te = df_test[dis_col].astype(int)
            if y_te.sum() < 5 or y_te.nunique() < 2:
                continue
            prob = mdl.predict_proba(X_te)[:, 1]
            auc_temporal = roc_auc_score(y_te, prob)
            X_all = df_g[X_FEATURES]
            y_all = df_g[dis_col].astype(int)
            _, X_val, _, y_val = train_test_split(
                X_all, y_all, test_size=0.3, random_state=SEED, stratify=y_all)
            auc_ho = roc_auc_score(y_val, mdl.predict_proba(X_val)[:, 1])
            gap = round(abs(auc_ho - auc_temporal), 4)
            temporal_rows.append({
                '집단': grp, '질환': dis_name, '알고리즘': algo,
                'HO_AUC': round(auc_ho, 4),
                'Temporal_AUC_2024': round(auc_temporal, 4),
                '|Gap|': gap,
                '판정': judge(gap, 'G2_temporal_gap'),
                '데이터한계_주석': '횡단면 구조상 단일연도(2024) 분할 검증; 코호트 연계 시 종단 확장 가능',
            })
    temporal_df = pd.DataFrame(temporal_rows)
    if len(temporal_df) > 0:
        print(temporal_df.to_string(index=False))
    else:
        print("  ⚠️ 2024년 데이터 부족 — temporal validation 결과 없음")
        temporal_df = pd.DataFrame()
else:
    temporal_df = pd.DataFrame()
    print("  ⚠️ SurveyYear 컬럼 없음 — temporal validation 불가")

print("\n" + "="*55)
print("[G2-c] 견고성: Feature Perturbation (±10% 노이즈)")
print("="*55)

perturb_rows = []
rng = np.random.RandomState(SEED)

for grp, cfg in AGEGROUP_CONFIG.items():
    df_g = df_final[
        (df_final['AgeGroup'] == cfg['age_group']) &
        (df_final['Sex']      == cfg['sex_code'])
    ].copy()
    for dis_col, dis_name in DISEASES.items():
        mdl  = best_models.get(grp, {}).get(dis_col)
        algo = best_algo_name.get(grp, {}).get(dis_col)
        if mdl is None:
            continue
        X = df_g[X_FEATURES]
        y = df_g[dis_col].astype(int)
        _, X_val, _, y_val = train_test_split(
            X, y, test_size=0.3, random_state=SEED, stratify=y)
        auc_base = roc_auc_score(y_val, mdl.predict_proba(X_val)[:, 1])
        drops = []
        for _ in range(5):
            X_noisy = X_val.copy()
            for col in X_noisy.columns:
                std = X_noisy[col].std()
                if std > 0:
                    noise = rng.normal(0, std * 0.10, len(X_noisy))
                    X_noisy[col] = X_noisy[col] + noise
            auc_noisy = roc_auc_score(y_val, mdl.predict_proba(X_noisy)[:, 1])
            drops.append(abs(auc_base - auc_noisy))
        mean_drop = round(np.mean(drops), 4)
        std_drop  = round(np.std(drops), 4)
        perturb_rows.append({
            '집단': grp, '질환': dis_name, '알고리즘': algo,
            'AUC_base': round(auc_base, 4),
            'AUC_drop_mean': mean_drop, 'AUC_drop_std': std_drop,
            '판정': judge(mean_drop, 'G2_perturb_drop'),
            '근거': '±10% 가우시안 노이즈 5회 반복',
        })

perturb_df = pd.DataFrame(perturb_rows)
print(perturb_df.to_string(index=False))


# %% =========================================================
# § G3  투명성: SHAP HHI + Bootstrap 95% CI
# ============================================================

def compute_hhi_bootstrap(shap_values, n_bootstrap=200, seed=42):
    rng = np.random.RandomState(seed)
    n = len(shap_values)
    hhis = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        sv_boot = shap_values[idx]
        mean_abs = np.abs(sv_boot).mean(axis=0)
        total = mean_abs.sum()
        if total == 0:
            continue
        hhi = float(((mean_abs / total) ** 2).sum())
        hhis.append(hhi)
    return np.percentile(hhis, 2.5), np.percentile(hhis, 97.5)

print("\n" + "="*55)
print("[G3] 투명성 지표: SHAP HHI + Bootstrap 95% CI")
print("="*55)

tra_rows = []
for grp in shap_results:
    for dis_col, dis_name in DISEASES.items():
        res = shap_results.get(grp, {}).get(dis_col)
        if res is None:
            continue
        sv       = res['shap_values']
        mean_abs = np.abs(sv).mean(axis=0)
        total    = mean_abs.sum()
        if total == 0:
            continue
        hhi = round(float(((mean_abs / total) ** 2).sum()), 4)
        ci_lo, ci_hi = compute_hhi_bootstrap(sv, 200, SEED)
        tra_rows.append({
            '집단': grp, '질환': dis_name,
            'SHAP_HHI': hhi,
            'HHI_95CI_lo': round(ci_lo, 4),
            'HHI_95CI_hi': round(ci_hi, 4),
            '투명성_판정': judge(hhi, 'G3_hhi'),
            '근거': THRESHOLDS['G3_hhi']['cite'],
        })

transparency_df = pd.DataFrame(tra_rows)
print(transparency_df.to_string(index=False))


# %% =========================================================
# § G4  설명책임: Model Card 체크리스트
# ============================================================

print("\n" + "="*55)
print("[G4] 설명책임 지표: Model Card 체크리스트")
print("="*55)

total_items = len(MODEL_CARD_CHECKLIST)
fulfilled   = sum(MODEL_CARD_CHECKLIST.values())
mc_score    = round(fulfilled / total_items, 2)

acc_rows = []
for grp in shap_results:
    for dis_col, dis_name in DISEASES.items():
        res = shap_results.get(grp, {}).get(dis_col)
        ok  = res is not None and res.get('shap_values') is not None
        acc_rows.append({
            '집단': grp, '질환': dis_name,
            'Waterfall생성': '가능 ✓' if ok else '불가 ✗',
            '코드공개': '공개 ✓' if GITHUB_URL else '미공개 ✗',
            'ModelCard_Score': mc_score,
            'ModelCard_판정': judge(mc_score, 'G4_model_card', low_best=False),
            '충족항목': f'{fulfilled}/{total_items}',
            '미충족항목': 'human-in-the-loop, PIA (향후 과제)',
            '근거': THRESHOLDS['G4_model_card']['cite'],
        })

accountability_df = pd.DataFrame(acc_rows)
print(accountability_df.to_string(index=False))

missing_items = [k for k, v in MODEL_CARD_CHECKLIST.items() if not v]
print(f"\n[미충족 항목 — 향후 과제]")
for item in missing_items:
    print(f"  ✗ {item}")
print(f"\n  Model Card Score: {mc_score} ({fulfilled}/{total_items})")


# %% =========================================================
# § 표6  G1–G4 거버넌스 정량 평가 종합
# ============================================================

gov_summary_rows = []

for dis_name in ['당뇨', '고혈압']:
    row_fair = fairness_summary[fairness_summary['질환'] == dis_name]
    row_rob  = robustness_df[robustness_df['질환'] == dis_name]
    row_per  = perturb_df[perturb_df['질환'] == dis_name] if len(perturb_df) > 0 else pd.DataFrame()
    row_tra  = transparency_df[transparency_df['질환'] == dis_name]
    row_temp = temporal_df[temporal_df['질환'] == dis_name] if len(temporal_df) > 0 else pd.DataFrame()

    gov_summary_rows.append({
        '질환': dis_name,
        '원칙': 'G1 공정성',
        '지표': 'Δ-AUC',
        '값': row_fair['Δ_AUC'].values[0] if len(row_fair) > 0 else '-',
        '기준': '< 0.10',
        '판정': row_fair['Δ_AUC_판정'].values[0] if len(row_fair) > 0 else '-',
        '규제근거': 'EU AI Act Art.10',
        '비고': '6집단 층화 완화 적용',
    })
    gov_summary_rows.append({
        '질환': dis_name,
        '원칙': 'G1 공정성',
        '지표': 'Equalized Odds Gap',
        '값': row_fair['EqOdds_Gap'].values[0] if len(row_fair) > 0 else '-',
        '기준': '< 0.15',
        '판정': row_fair['EqOdds_판정'].values[0] if len(row_fair) > 0 else '-',
        '규제근거': '금융위 AI 가이드라인',
        '비고': '6집단 층화 완화 적용',
    })
    gov_summary_rows.append({
        '질환': dis_name,
        '원칙': 'G1 공정성',
        '지표': 'ECE Gap',
        '값': row_fair['Δ_ECE'].values[0] if len(row_fair) > 0 else '-',
        '기준': '< 0.05',
        '판정': row_fair['ECE_판정'].values[0] if len(row_fair) > 0 else '-',
        '규제근거': '의료기기 소프트웨어 가이드라인',
        '비고': '',
    })
    gov_summary_rows.append({
        '질환': dis_name,
        '원칙': 'G2 견고성',
        '지표': 'CV vs HO Gap',
        '값': round(row_rob['|Gap|'].mean(), 4) if len(row_rob) > 0 else '-',
        '기준': '< 0.03',
        '판정': judge(row_rob['|Gap|'].mean(), 'G2_cv_gap') if len(row_rob) > 0 else '-',
        '규제근거': 'OECD AI 원칙',
        '비고': '집단 평균값',
    })
    gov_summary_rows.append({
        '질환': dis_name,
        '원칙': 'G2 견고성',
        '지표': 'Temporal Gap (2024)',
        '값': round(row_temp['|Gap|'].mean(), 4) if len(row_temp) > 0 else '데이터부족',
        '기준': '< 0.05',
        '판정': judge(row_temp['|Gap|'].mean(), 'G2_temporal_gap') if len(row_temp) > 0 else '해당없음',
        '규제근거': 'OECD AI 원칙',
        '비고': '횡단면 구조 — 단일연도 분할 한계',
    })
    gov_summary_rows.append({
        '질환': dis_name,
        '원칙': 'G2 견고성',
        '지표': 'Perturbation AUC Drop',
        '값': round(row_per['AUC_drop_mean'].mean(), 4) if len(row_per) > 0 else '-',
        '기준': '< 0.03',
        '판정': judge(row_per['AUC_drop_mean'].mean(), 'G2_perturb_drop') if len(row_per) > 0 else '-',
        '규제근거': 'OECD AI 원칙',
        '비고': '±10% 가우시안 노이즈 5회 평균',
    })
    gov_summary_rows.append({
        '질환': dis_name,
        '원칙': 'G3 투명성',
        '지표': 'SHAP HHI (Bootstrap 95% CI)',
        '값': f"{round(row_tra['SHAP_HHI'].mean(),4)} [{round(row_tra['HHI_95CI_lo'].mean(),4)}–{round(row_tra['HHI_95CI_hi'].mean(),4)}]" if len(row_tra) > 0 else '-',
        '기준': '< 0.15',
        '판정': judge(row_tra['SHAP_HHI'].mean(), 'G3_hhi') if len(row_tra) > 0 else '-',
        '규제근거': '개보법 제37조의2',
        '비고': 'HHI 낮을수록 설명 분산 — 투명성 높음',
    })
    gov_summary_rows.append({
        '질환': dis_name,
        '원칙': 'G4 설명책임',
        '지표': 'Model Card Score',
        '값': mc_score,
        '기준': '≥ 0.80',
        '판정': judge(mc_score, 'G4_model_card', low_best=False),
        '규제근거': 'EU AI Act Art.13',
        '비고': f'{fulfilled}/{total_items} 충족; 미충족: human-in-the-loop, PIA',
    })

table6 = pd.DataFrame(gov_summary_rows)
table6.to_csv('표6_거버넌스정량평가종합.csv', index=False, encoding='utf-8-sig')
print("\n>>> 저장: 표6_거버넌스정량평가종합.csv")
print(table6.to_string(index=False))


# %% =========================================================
# § 표7  이해관계자 역할 정의
# ============================================================

table7 = pd.DataFrame([
    {
        '역할': '데이터 과학팀 (개발·갱신)',
        '주체': '건강검진 기관 내 AI 개발팀',
        '책임': '연간 모델 재학습, SHAP 결과 갱신, Model Card 업데이트',
        '트리거': 'Δ-AUC > 0.10 또는 신규 연도 KNHANES 공개',
        '산출물': '갱신된 모델 파일, 성능 비교 보고서',
    },
    {
        '역할': '공정성 감사 (내부 AI 윤리위)',
        '주체': '기관 내 AI 윤리위원회 또는 외부 감사기관',
        '책임': 'G1 공정성 지표 반기 검토, EqOdds Gap 모니터링',
        '트리거': '반기별 정기 감사 또는 민원 발생 시',
        '산출물': '공정성 감사 보고서, 개선 권고안',
    },
    {
        '역할': '정책 결정 (기관장·감독부서)',
        '주체': '기관장, 보건부, 건강보험심사평가원',
        '책임': '등급 기준 변경 승인, 거버넌스 프레임워크 갱신 결정',
        '트리거': '공정성 감사 결과 위험 판정 또는 연 1회 정기 검토',
        '산출물': '정책 결정문, 등급 기준 고시',
    },
    {
        '역할': '이용자 권리 (피검자)',
        '주체': '건강검진 수검자',
        '책임': '등급 결과 이의신청, SHAP 기반 설명 요구',
        '트리거': '등급 결과 수령 후 30일 이내',
        '산출물': '이의신청서, 개인별 SHAP Waterfall 설명서',
    },
    {
        '역할': '감독기관 (외부)',
        '주체': '개인정보보호위원회, 보건복지부',
        '책임': 'EU AI Act·개보법 준수 여부 확인, PIA 검토',
        '트리거': '연 1회 정기 감사 또는 사안 발생 시',
        '산출물': '준수 확인서, 시정명령',
    },
])
table7.to_csv('표7_이해관계자역할정의.csv', index=False, encoding='utf-8-sig')
print(">>> 저장: 표7_이해관계자역할정의.csv")
print(table7.to_string(index=False))


# %% =========================================================
# § 표8  거버넌스 운영 사이클 단계
# ============================================================

table8 = pd.DataFrame([
    {
        '단계': 1,
        '단계명': '데이터 갱신',
        '주기': '연 1회 (신규 KNHANES 공개 시)',
        '담당': '데이터 과학팀',
        '활동': '신규 연도 KNHANES 수집·전처리, 집단별 유병률 변화 모니터링',
        '판단기준': '유병자 수 50명 이상 여부 확인',
        '재학습트리거': '신규 연도 데이터 공개',
    },
    {
        '단계': 2,
        '단계명': 'G1 공정성 재측정',
        '주기': '연 1회 (데이터 갱신 후)',
        '담당': '데이터 과학팀 + AI 윤리위',
        '활동': 'Δ-AUC, EqOdds Gap, ECE Gap 재산출 및 판정',
        '판단기준': 'Δ-AUC > 0.10 또는 EqOdds > 0.15',
        '재학습트리거': '기준 초과 시 → 집단별 재학습 및 하이퍼파라미터 재최적화',
    },
    {
        '단계': 3,
        '단계명': 'G2 견고성 재측정',
        '주기': '연 1회',
        '담당': '데이터 과학팀',
        '활동': 'CV Gap, Temporal Gap(최신연도), Perturbation Drop 재산출',
        '판단기준': 'CV Gap > 0.03 또는 Perturbation Drop > 0.03',
        '재학습트리거': '기준 초과 시 → Optuna 재최적화 (n_trials 증가)',
    },
    {
        '단계': 4,
        '단계명': 'G3 투명성 재측정',
        '주기': '연 1회',
        '담당': '데이터 과학팀',
        '활동': 'SHAP HHI Bootstrap 재산출, 상위 변수 변동 모니터링',
        '판단기준': 'HHI > 0.15 또는 주요 변수 순위 급변',
        '재학습트리거': '이상 감지 시 → 피처 중요도 재검토',
    },
    {
        '단계': 5,
        '단계명': 'G4 Model Card 갱신',
        '주기': '연 1회',
        '담당': '데이터 과학팀 + 기관장',
        '활동': 'Model Card 10개 항목 재점검, GitHub 코드 갱신',
        '판단기준': 'Score < 0.80',
        '재학습트리거': '미충족 항목 개선 계획 수립 및 이행',
    },
    {
        '단계': 6,
        '단계명': '이해관계자 보고 및 결정',
        '주기': '연 1회',
        '담당': '기관장·감독부서',
        '활동': 'G1–G4 종합 보고, 운영 지속 or 모델 교체 결정, 등급 기준 검토',
        '판단기준': '가중종합 거버넌스 점수 ≥ 0.75 (양호)',
        '재학습트리거': '점수 < 0.60 시 전면 재설계 검토',
    },
])
table8.to_csv('표8_거버넌스운영사이클.csv', index=False, encoding='utf-8-sig')
print(">>> 저장: 표8_거버넌스운영사이클.csv")
print(table8.to_string(index=False))


# %% =========================================================
# § 거버넌스 종합 CSV (기존 호환 유지)
# ============================================================

gov_merged = (df_fair
    .merge(robustness_df[['집단', '질환', '|Gap|', '견고성_판정']],
           on=['집단', '질환'], how='left')
    .merge(transparency_df[['집단', '질환', 'SHAP_HHI', 'HHI_95CI_lo', 'HHI_95CI_hi', '투명성_판정']],
           on=['집단', '질환'], how='left')
    .merge(accountability_df[['집단', '질환', 'Waterfall생성', '코드공개', 'ModelCard_Score']],
           on=['집단', '질환'], how='left')
)
if len(perturb_df) > 0:
    gov_merged = gov_merged.merge(
        perturb_df[['집단', '질환', 'AUC_drop_mean', '판정']].rename(
            columns={'판정': 'Perturbation_판정'}),
        on=['집단', '질환'], how='left')
if len(temporal_df) > 0:
    gov_merged = gov_merged.merge(
        temporal_df[['집단', '질환', 'Temporal_AUC_2024', '판정']].rename(
            columns={'판정': 'Temporal_판정'}),
        on=['집단', '질환'], how='left')

gov_merged.to_csv('거버넌스_지표_종합_v2.1.csv', index=False, encoding='utf-8-sig')
table5.to_csv('거버넌스_원칙정의_v2.1.csv', index=False, encoding='utf-8-sig')
print("\n>>> 저장: 거버넌스_원칙정의_v2.1.csv / 거버넌스_지표_종합_v2.1.csv")


# %% =========================================================
# § 그림 2–12
# ============================================================

# ── [그림 2] 집단별 유병률 (2×3) ──
def fig2_prevalence(save_path='그림2_집단별유병률.png'):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=False)
    axes = axes.flatten()
    for ax, grp, cfg, color, lbl in zip(
            axes, GRP_NAMES, AGEGROUP_CONFIG.values(), COLORS, GRP_LABEL):
        df_g = df_final[(df_final['AgeGroup'] == cfg['age_group']) &
                         (df_final['Sex']      == cfg['sex_code'])]
        dm_r = df_g['Diabetes'].mean() * 100
        hp_r = df_g['Hypertension'].mean() * 100
        n    = len(df_g)
        b1 = ax.bar(['당뇨'],   [dm_r], color=color, alpha=1.0,
                    edgecolor='white', linewidth=0.8, width=0.45)
        b2 = ax.bar(['고혈압'], [hp_r], color=color, alpha=0.55,
                    edgecolor='white', linewidth=0.8, width=0.45)
        for bar, val in zip(list(b1) + list(b2), [dm_r, hp_r]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5, f'{val:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_title(f'{lbl}  (n={n:,})', fontsize=10, pad=5)
        ax.set_ylim(0, 80)
        ax.set_ylabel('유병률 (%)', fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"저장: {save_path}")

# ── [그림 3] AUC 히트맵 (LR Baseline 포함) ──
def fig3_auc_heatmap(save_path='그림3_AUC히트맵.png'):
    rows, labels = [], []
    for grp, lbl in zip(GRP_NAMES, GRP_LABEL):
        for dis, dis_lbl in [('Diabetes', '당뇨'), ('Hypertension', '고혈압')]:
            aucs = [
                (all_results.get(grp, {}).get(dis, {}).get(algo) or {}).get('AUC', np.nan)
                for algo in ALGOS
            ]
            rows.append(aucs)
            labels.append(f'{lbl} ({dis_lbl})')
    data = np.array(rows)
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(data, cmap='RdYlGn', vmin=0.50, vmax=0.90, aspect='auto')
    plt.colorbar(im, ax=ax, shrink=0.7).set_label('AUC', fontsize=10)
    ax.set_xticks(range(len(ALGOS)))
    ax.set_xticklabels([f'{a}\n(Baseline)' if a == 'LR' else a for a in ALGOS], fontsize=10)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(labels)):
        # LR 제외한 최적
        non_lr = [j for j, a in enumerate(ALGOS) if a != 'LR']
        best_j = non_lr[int(np.nanargmax([data[i, j] for j in non_lr]))]
        for j in range(len(ALGOS)):
            val = data[i, j]
            if np.isnan(val): continue
            txt_c  = 'white' if (val < 0.62 or val > 0.82) else 'black'
            weight = 'bold' if j == best_j else 'normal'
            marker = '★' if j == best_j else ('BL' if ALGOS[j] == 'LR' else '')
            ax.text(j, i, f'{val:.3f}\n{marker}', ha='center', va='center',
                    fontsize=8, color=txt_c, fontweight=weight)
    ax.set_title('집단별·알고리즘별 AUC 비교 (★: 최적, BL: Baseline)', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"저장: {save_path}")

# ── [그림 4] 엘보우 곡선 (2×3) ──
def fig4_elbow(save_path='그림4_엘보우곡선.png'):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for ax, grp, cfg, color, lbl in zip(
            axes, GRP_NAMES, AGEGROUP_CONFIG.values(), COLORS, GRP_LABEL):
        df_g   = df_final[(df_final['AgeGroup'] == cfg['age_group']) &
                           (df_final['Sex']      == cfg['sex_code'])].copy()
        mdl_dm = best_models.get(grp, {}).get('Diabetes')
        mdl_hp = best_models.get(grp, {}).get('Hypertension')
        if mdl_dm is None or mdl_hp is None:
            ax.set_title(lbl); ax.axis('off'); continue
        X       = df_g[X_FEATURES]
        prob_dm = mdl_dm.predict_proba(X)[:, 1]
        prob_hp = mdl_hp.predict_proba(X)[:, 1]
        X_km    = np.column_stack([prob_dm, prob_hp])
        k_rng   = range(2, 8)
        inerts  = [KMeans(n_clusters=k, random_state=SEED, n_init=10)
                    .fit(X_km).inertia_ for k in k_rng]
        ax.plot(list(k_rng), inerts, 'o-', color=color, lw=2, ms=6)
        ax.axvline(x=3, color='red', linestyle='--', lw=1.2, alpha=0.8)
        ax.text(3.1, max(inerts) * 0.95, 'K=3', color='red', fontsize=9)
        ax.set_xlabel('군집 수 K', fontsize=9)
        ax.set_ylabel('관성 (Inertia)', fontsize=9)
        ax.set_title(lbl, fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"저장: {save_path}")

# ── [그림 5] 복합건강등급 분포 (2×3) ──
def fig5_grade_dist(save_path='그림5_복합건강등급분포.png'):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for ax, grp, color, lbl in zip(axes, GRP_NAMES, COLORS, GRP_LABEL):
        df_g = grade_results.get(grp)
        if df_g is None:
            ax.set_title(lbl); ax.axis('off'); continue
        summary = df_g.groupby('복합건강등급').agg(
            건수=('복합건강등급', 'count'),
            당뇨유병률=('Diabetes', 'mean'),
            고혈압유병률=('Hypertension', 'mean'),
        )
        summary['구성비'] = summary['건수'] / summary['건수'].sum() * 100
        grades = summary.index.tolist()
        ax2 = ax.twinx()
        ax.bar(grades, summary['구성비'], color=color, alpha=0.4,
               label='구성비', zorder=2)
        ax2.plot(grades, summary['당뇨유병률'] * 100,
                 'o-r', lw=2, ms=6, label='당뇨 유병률', zorder=3)
        ax2.plot(grades, summary['고혈압유병률'] * 100,
                 's--b', lw=2, ms=6, label='고혈압 유병률', zorder=3)
        for g, dm, hp in zip(grades,
                              summary['당뇨유병률'] * 100,
                              summary['고혈압유병률'] * 100):
            ax2.text(g, dm + 2, f'{dm:.1f}%', ha='center', fontsize=7, color='red')
            ax2.text(g, max(hp - 7, 1), f'{hp:.1f}%', ha='center', fontsize=7, color='blue')
        ax.set_xlabel('복합건강등급', fontsize=9)
        ax.set_ylabel('구성비 (%)', fontsize=9, color=color)
        ax2.set_ylabel('유병률 (%)', fontsize=9)
        ax2.set_ylim(0, 105); ax.set_ylim(0, 70)
        ax.set_title(lbl, fontsize=10)
        ax.set_xticks(grades)
        ax.set_xticklabels(
            ['1등급\n(저위험)', '2등급\n(중위험)', '3등급\n(고위험)'], fontsize=8)
        ax.spines[['top']].set_visible(False)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"저장: {save_path}")

# ── [그림 6·7] SHAP Summary (2×3) ──
def fig6_7_shap_summary(disease, dis_label, fig_num,
                         save_path_tpl='그림{n}_SHAP_Summary_{d}.png'):
    fig = plt.figure(figsize=(13, 26))          # ← 가로 줄이고 세로 늘림
    gs  = fig.add_gridspec(3, 2,                # ← 3행 2열 (기존 2행 3열 → 반전)
                            left=0.08, right=0.84,
                            top=0.93, bottom=0.06,
                            wspace=0.55,         # ← 좌우 간격
                            hspace=0.55)         # ← 상하 간격

    for idx, (grp, lbl) in enumerate(zip(GRP_NAMES, GRP_LABEL)):
        row = idx // 2                           # ← 2열 기준 행 계산
        col = idx % 2                            # ← 2열 기준 열 계산
        ax  = fig.add_subplot(gs[row, col])

        res = shap_results.get(grp, {}).get(disease)
        if res is None:
            ax.set_title(lbl, fontsize=10); ax.axis('off'); continue

        plt.sca(ax)
        shap.summary_plot(res['shap_values'], res['X_sample'],
                          plot_type='dot', show=False,
                          max_display=12,
                          color_bar=False)

        ax.set_title(lbl, fontsize=10, pad=5)
        ax.tick_params(axis='x', labelsize=7)
        ax.tick_params(axis='y', labelsize=7)

        for extra_ax in fig.axes:
            if extra_ax is ax:
                continue
            if extra_ax.get_position().width < 0.05:
                extra_ax.remove()

    # 공통 colorbar
    cbar_ax = fig.add_axes([0.86, 0.20, 0.015, 0.55])
    sm = plt.cm.ScalarMappable(
        cmap=shap.plots.colors.red_blue,
        norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('특성값', fontsize=9, labelpad=8, rotation=270, va='bottom')
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['낮음', '중간', '높음'], fontsize=8)

    fname = save_path_tpl.format(n=fig_num, d=dis_label)
    plt.savefig(fname, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"저장: {fname}")

# ── [그림 8] SHAP 집단 간 변수 중요도 비교 ──
def fig8_shap_comparison(top_n=10, save_path='그림8_SHAP집단비교.png'):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, disease, dis_lbl in zip(
            axes, ['Diabetes', 'Hypertension'], ['당뇨', '고혈압']):
        imp_dict = {}
        for grp, lbl in zip(GRP_NAMES, GRP_LABEL):
            res = shap_results.get(grp, {}).get(disease)
            if res is None: continue
            sv   = res['shap_values']
            cols = res['X_sample'].columns.tolist()
            imp_dict[lbl] = pd.Series(np.abs(sv).mean(axis=0), index=cols)
        if not imp_dict: continue
        all_imp   = pd.DataFrame(imp_dict).fillna(0)
        top_feats = all_imp.mean(axis=1).nlargest(top_n).index.tolist()
        plot_df   = all_imp.loc[top_feats]
        x, width  = np.arange(len(top_feats)), 0.13
        for i, (col, color) in enumerate(zip(plot_df.columns, COLORS)):
            ax.barh(x + i * width, plot_df[col].values,
                    height=width, label=col, color=color, alpha=0.85)
        ax.set_yticks(x + width * 2.5)
        ax.set_yticklabels(top_feats, fontsize=8)
        ax.set_xlabel('SHAP 절대값 평균', fontsize=10)
        ax.set_title(f'{dis_lbl} — 집단별 변수 중요도', fontsize=11)
        ax.legend(fontsize=8, loc='lower right')
        ax.spines[['top', 'right']].set_visible(False)
        ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"저장: {save_path}")

# ── [그림 9] 공정성 지표 (AUC + EqOdds) ──
def fig9_fairness(save_path='그림9_공정성지표_v2.1.png'):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for ax, dis_name in zip(axes[0], ['당뇨', '고혈압']):
        sub = df_fair[df_fair['질환'] == dis_name]
        auc_vals = [
            sub[sub['집단'] == g]['AUC'].values[0]
            if len(sub[sub['집단'] == g]) > 0 else 0
            for g in GRP_NAMES
        ]
        bars = ax.bar(GRP_LABEL, auc_vals, color=COLORS,
                      alpha=0.85, edgecolor='white', linewidth=0.8)
        for bar, val in zip(bars, auc_vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003, f'{val:.3f}',
                    ha='center', va='bottom', fontsize=8)
        auc_arr = np.array(auc_vals)
        valid = auc_arr[auc_arr > 0]
        if len(valid) >= 2:
            auc_min, auc_max = valid.min(), valid.max()
            delta = round(auc_max - auc_min, 4)
            row = fairness_summary[fairness_summary['질환'] == dis_name]
            status = row['Δ_AUC_판정'].values[0] if len(row) > 0 else ''
            ax.axhline(auc_min, color='red',  linestyle='--', lw=1.2, alpha=0.7)
            ax.axhline(auc_max, color='navy', linestyle='--', lw=1.2, alpha=0.7)
            ax.annotate('', xy=(5.6, auc_max), xytext=(5.6, auc_min),
                        arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
            ax.text(5.7, (auc_min + auc_max) / 2,
                    f'Δ={delta:.4f}\n{status}',
                    fontsize=9, color='red', va='center')
            ax.set_ylim(max(0, auc_min - 0.05), min(1.0, auc_max + 0.08))
        ax.set_ylabel('AUC', fontsize=11)
        ax.set_title(f'{dis_name} — 집단별 AUC 공정성 (6집단 층화)', fontsize=12)
        ax.tick_params(axis='x', labelsize=8, rotation=15)
        ax.spines[['top', 'right']].set_visible(False)
    for ax, dis_name in zip(axes[1], ['당뇨', '고혈압']):
        sub = df_fair[df_fair['질환'] == dis_name]
        if len(sub) == 0: continue
        x = np.arange(len(GRP_LABEL))
        width = 0.35
        tpr_vals = [sub[sub['집단'] == g]['TPR'].values[0]
                    if len(sub[sub['집단'] == g]) > 0 else 0 for g in GRP_NAMES]
        fpr_vals = [sub[sub['집단'] == g]['FPR'].values[0]
                    if len(sub[sub['집단'] == g]) > 0 else 0 for g in GRP_NAMES]
        ax.bar(x - width/2, tpr_vals, width, label='TPR (민감도)', color='#2196F3', alpha=0.8)
        ax.bar(x + width/2, fpr_vals, width, label='FPR (위양성률)', color='#E91E63', alpha=0.8)
        for i, (t, f) in enumerate(zip(tpr_vals, fpr_vals)):
            ax.text(i - width/2, t + 0.01, f'{t:.2f}', ha='center', fontsize=7)
            ax.text(i + width/2, f + 0.01, f'{f:.2f}', ha='center', fontsize=7)
        row = fairness_summary[fairness_summary['질환'] == dis_name]
        if len(row) > 0:
            eq_gap = row['EqOdds_Gap'].values[0]
            eq_status = row['EqOdds_판정'].values[0]
            ax.text(0.98, 0.99, f'EqOdds Gap = {eq_gap:.4f}  {eq_status}',
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                              edgecolor='orange', alpha=0.95),
                    zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels(GRP_LABEL, fontsize=8, rotation=15)
        ax.set_ylabel('비율', fontsize=10)
        ax.set_title(f'{dis_name} — Equalized Odds (TPR/FPR)', fontsize=12)
        ax.legend(fontsize=8,
            loc='upper right',
            bbox_to_anchor=(1.0, 0.88),   # EqOdds 텍스트 박스 아래로 내림
            framealpha=0.9,
            edgecolor='#CCCCCC',
        )
        ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"저장: {save_path}")

# ── [그림 10] 거버넌스 레이더 (6축 + 가중종합) ──
def fig10_governance_radar_v2(save_path='그림10_거버넌스레이더_v2.1.png'):

    weights = {
        'G1a\nΔ-AUC':   0.20,
        'G1b\nEqOdds':  0.15,
        'G2a\nCV-Gap':  0.15,
        'G2b\nPerturb': 0.10,
        'G3\nHHI':      0.20,
        'G4\n설명책임':  0.20,
    }

    categories = list(weights.keys())
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, 2, figsize=(20, 10),
                             subplot_kw=dict(polar=True))
    composite_scores = {}

    for ax, dis_name, color in zip(
            axes, ['당뇨', '고혈압'], ['#2196F3', '#E91E63']):

        row     = fairness_summary[fairness_summary['질환'] == dis_name]
        rob_sub = robustness_df[robustness_df['질환'] == dis_name]
        tra_sub = transparency_df[transparency_df['질환'] == dis_name]
        per_sub = (perturb_df[perturb_df['질환'] == dis_name]
                   if len(perturb_df) > 0 else pd.DataFrame())

        # ── 실제 지표값 ──
        delta_auc_val = row['Δ_AUC'].values[0]         if len(row)     > 0 else 0.15
        eq_val        = row['EqOdds_Gap'].values[0]    if len(row)     > 0 else 0.20
        cv_gap_val    = rob_sub['|Gap|'].mean()         if len(rob_sub) > 0 else 0.04
        perturb_val   = (per_sub['AUC_drop_mean'].mean()
                         if len(per_sub) > 0 else 0.04)
        hhi_val       = tra_sub['SHAP_HHI'].mean()     if len(tra_sub) > 0 else 0.18

        # ── 정규화 점수 ──
        g1a = norm_score(delta_auc_val, good=0.10, bad=0.20, floor=0.10)
        g1b = norm_score(eq_val,        good=0.15, bad=0.30, floor=0.10)
        g2a = norm_score(cv_gap_val,    good=0.03, bad=0.06, floor=0.10)
        g2b = norm_score(perturb_val,   good=0.03, bad=0.05, floor=0.10)
        g3  = norm_score(hhi_val,       good=0.15, bad=0.25, floor=0.10)
        g4  = min(1.0, mc_score)

        scores        = [g1a, g1b, g2a, g2b, g3, g4]
        scores_closed = scores + [scores[0]]

        w_vals    = list(weights.values())
        composite = round(sum(s * w for s, w in zip(scores, w_vals)), 2)
        composite_scores[dis_name] = composite

        # ── 레이더 본체 ──
        ax.plot(angles, scores_closed, 'o-', color=color,
                lw=2.5, ms=7, zorder=3)
        ax.fill(angles, scores_closed, color=color, alpha=0.20, zorder=2)

        # 기준선
        ax.plot(angles, [0.8] * (N + 1), '--', color='green',
                lw=1.2, alpha=0.6, label='양호 기준 (0.8)')
        ax.plot(angles, [0.6] * (N + 1), ':',  color='orange',
                lw=1.2, alpha=0.6, label='주의 기준 (0.6)')

        # ── 축 설정 ──
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([])          # 기본 레이블 제거 → 수동 배치
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'],
                           fontsize=7, color='gray')
        ax.tick_params(pad=5)

        # ── 축 레이블 수동 배치 (G2b만 더 바깥) ──
        label_offsets = [1.22, 1.22, 1.22, 1.32, 1.22, 1.22]
        for angle, cat, offset in zip(angles[:-1], categories, label_offsets):
            ax.text(angle, offset, cat,
                    ha='center', va='center',
                    fontsize=10, fontweight='bold',
                    transform=ax.transData)

        # ── 꼭짓점 수치 레이블 ──
        raw_vals = [delta_auc_val, eq_val, cv_gap_val,
                    perturb_val,   hhi_val, mc_score]
        raw_lbls = ['Δ={:.3f}', 'EO={:.3f}', 'CV={:.3f}',
                    'Pt={:.3f}', 'HHI={:.3f}', 'MC={:.2f}']

        for angle, score, raw, fmt in zip(
                angles[:-1], scores, raw_vals, raw_lbls):
            # 값이 높으면(0.85↑) 안쪽, 낮으면 바깥쪽에 배치
            if score >= 0.85:
                label_r = score - 0.20
            else:
                label_r = score + 0.17
            label_r = max(0.12, min(label_r, 0.92))   # 범위 클램핑

            ax.text(angle, label_r,
                    f'{score:.2f}\n({fmt.format(raw)})',
                    ha='center', va='center',
                    fontsize=7.5, color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.18',
                              facecolor='white',
                              edgecolor=color, alpha=0.88),
                    zorder=5)

        # ── 제목 ──
        grade_txt = ('[양호]' if composite >= 0.75
                     else '[주의]' if composite >= 0.60
                     else '[위험]')
        ax.set_title(f'{dis_name}\n가중종합 = {composite:.2f} {grade_txt}',
                     fontsize=13, fontweight='bold', pad=40)

        # ── 범례 ──
        ax.legend(fontsize=8, loc='lower center',
                  bbox_to_anchor=(0.5, -0.18),
                  ncol=2, framealpha=0.9)

    plt.tight_layout(pad=4.0)
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"저장: {save_path}")
    print(f"  당뇨  가중종합: {composite_scores.get('당뇨', '-')}")
    print(f"  고혈압 가중종합: {composite_scores.get('고혈압', '-')}")

# ── [그림 11] Perturbation 견고성 히트맵 ──
def fig11_perturbation_heatmap(save_path='그림11_Perturbation견고성.png'):
    if len(perturb_df) == 0:
        print("  ⚠️ Perturbation 결과 없음 — 그림 11 생략")
        return
    rows_dm, rows_hp = [], []
    for grp in GRP_NAMES:
        for dis, container in [('당뇨', rows_dm), ('고혈압', rows_hp)]:
            sub = perturb_df[(perturb_df['집단'] == grp) & (perturb_df['질환'] == dis)]
            container.append(sub['AUC_drop_mean'].values[0] if len(sub) > 0 else np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, title in zip(axes,
            [rows_dm, rows_hp],
            ['당뇨 — AUC Drop (±10% 노이즈)', '고혈압 — AUC Drop (±10% 노이즈)']):
        vals = np.array(data).reshape(1, -1)
        im = ax.imshow(vals, cmap='RdYlGn_r', vmin=0, vmax=0.08, aspect='auto')
        ax.set_xticks(range(len(GRP_LABEL)))
        ax.set_xticklabels(GRP_LABEL, fontsize=8, rotation=15)
        ax.set_yticks([0])
        ax.set_yticklabels(['±10% 노이즈'], fontsize=9)
        for j, v in enumerate(data):
            if np.isnan(v): continue
            txt_c = 'white' if v > 0.04 else 'black'
            status = 'O' if v < 0.03 else '-' if v < 0.05 else 'X'
            ax.text(j, 0, f'{v:.4f}\n{status}', ha='center', va='center',
                    fontsize=9, fontweight='bold', color=txt_c)
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.6).set_label('AUC Drop', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"저장: {save_path}")

# ── [그림 12] 거버넌스 운영 사이클 플로우차트 (신규) ──
def fig12_governance_cycle(save_path='그림12_거버넌스운영사이클.png'):
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('#F8F9FA')

    # 박스 정의: (x중심, y중심, 너비, 높이, 색, 텍스트)
    boxes = [
        (2.0, 7.5, 3.2, 1.0, '#1565C0', '① 데이터 갱신\n(연 1회 KNHANES 공개)'),
        (7.0, 7.5, 3.2, 1.0, '#E65100', '② G1 공정성 재측정\nΔ-AUC / EqOdds / ECE'),
        (12.0, 7.5, 3.2, 1.0, '#2E7D32', '③ G2 견고성 재측정\nCV Gap / Temporal / Perturb'),
        (12.0, 4.5, 3.2, 1.0, '#6A1B9A', '④ G3 투명성 재측정\nSHAP HHI (Bootstrap CI)'),
        (7.0, 4.5, 3.2, 1.0, '#C62828', '⑤ G4 설명책임 갱신\nModel Card 재점검'),
        (2.0, 4.5, 3.2, 1.0, '#00695C', '⑥ 이해관계자 보고\n운영 지속 or 모델 교체'),
    ]

    box_centers = {}
    for (cx, cy, w, h, color, label) in boxes:
        fancy = FancyBboxPatch(
            (cx - w/2, cy - h/2), w, h,
            boxstyle='round,pad=0.08',
            facecolor=color, edgecolor='white',
            linewidth=2, alpha=0.92,
            zorder=3,
        )
        ax.add_patch(fancy)
        ax.text(cx, cy, label, ha='center', va='center',
                fontsize=9, color='white', fontweight='bold',
                zorder=4, multialignment='center')
        box_centers[label[:2]] = (cx, cy)

    # 재학습 트리거 박스
    triggers = [
        (5.0, 7.5, '#FF8F00', 'Δ-AUC>0.10\n→ 재학습'),
        (10.0, 7.5, '#558B2F', 'CV Gap>0.03\n→ 재최적화'),
        (10.0, 4.5, '#4527A0', 'HHI>0.15\n→ 피처 검토'),
        (5.0, 4.5, '#B71C1C', 'Score<0.80\n→ 항목 보완'),
    ]
    for (cx, cy, color, label) in triggers:
        fancy = FancyBboxPatch(
            (cx - 1.3, cy - 0.38), 2.6, 0.76,
            boxstyle='round,pad=0.05',
            facecolor=color, edgecolor='white',
            linewidth=1.5, alpha=0.75, linestyle='--',
            zorder=3,
        )
        ax.add_patch(fancy)
        ax.text(cx, cy, label, ha='center', va='center',
                fontsize=7.5, color='white', fontweight='bold',
                zorder=4, multialignment='center')

    # 화살표 (순환)
    arrow_style = dict(arrowstyle='->', color='#37474F', lw=2.0,
                       connectionstyle='arc3,rad=0.0')
    arrow_pairs = [
        ((3.6, 7.5), (5.4, 7.5)),    # ①→트리거1
        ((6.35, 7.5), (5.45, 7.5)),  # 트리거1←②
        ((8.6, 7.5), (10.35, 7.5)),  # ②→트리거2
        ((11.35, 7.5), (10.65, 7.5)),# 트리거2←③
        ((12.0, 7.0), (12.0, 5.0)),  # ③↓④
        ((11.35, 4.5), (10.65, 4.5)),# ④→트리거3
        ((9.35, 4.5), (8.3, 4.5)),   # 트리거3←⑤
        ((5.35, 4.5), (5.65, 4.5)),  # ⑤→트리거4 (반대)
        ((3.65, 4.5), (4.35, 4.5)),  # 트리거4→⑥
        ((2.0, 5.0), (2.0, 7.0)),    # ⑥↑① (순환)
    ]
    for (start, end) in arrow_pairs:
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=arrow_style, zorder=2)

    # 범례 박스
    legend_items = [
        ('#37474F', '→ 단계 흐름'),
        ('#FF8F00', '--- 재학습/재최적화 트리거'),
    ]
    for i, (color, label) in enumerate(legend_items):
        ax.text(0.5, 2.5 - i * 0.5, f'■ {label}',
                fontsize=8, color=color, va='center')

    # 데이터 한계 주석
    ax.text(7.0, 1.5,
            '※ Temporal Validation: KNHANES 횡단면 구조상 단일연도(2024) 분할 검증\n'
            '   (동일 개인 종단 추적 불가) — 향후 코호트 연계 시 롤링 검증으로 확장 가능',
            ha='center', va='center', fontsize=8,
            color='#546E7A', style='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#ECEFF1',
                      edgecolor='#90A4AE', alpha=0.9))

    ax.set_title('그림 12. AI 거버넌스 운영 사이클\n'
                 '(공정성·견고성·투명성·설명책임 기반 연간 모니터링 체계)',
                 fontsize=12, fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"저장: {save_path}")


# ── 일괄 실행 ──
print("\n" + "="*50)
print("그림 2–12 생성 시작")
print("="*50)

fig2_prevalence()
fig3_auc_heatmap()
fig4_elbow()
fig5_grade_dist()
fig6_7_shap_summary('Diabetes',     '당뇨',  6)
fig6_7_shap_summary('Hypertension', '고혈압', 7)
fig8_shap_comparison()
fig9_fairness()
fig10_governance_radar_v2()
fig11_perturbation_heatmap()
fig12_governance_cycle()

print("\n" + "="*60)
print(">>> Part 2 완료 — 생성 파일 목록")
print("="*60)
files = [
    # 표
    "표5_거버넌스원칙정의.csv         ← G1–G4 임계값 + 도메인적합성 주석",
    "표6_거버넌스정량평가종합.csv      ← G1–G4 수치 + 판정 + 규제근거 통합",
    "표7_이해관계자역할정의.csv        ← 역할·주체·책임·트리거·산출물",
    "표8_거버넌스운영사이클.csv        ← 6단계 운영절차 상세",
    "거버넌스_원칙정의_v2.1.csv",
    "거버넌스_지표_종합_v2.1.csv",
    # 그림
    "그림2_집단별유병률.png",
    "그림3_AUC히트맵.png              ← LR Baseline 포함",
    "그림4_엘보우곡선.png",
    "그림5_복합건강등급분포.png",
    "그림6_SHAP_Summary_당뇨.png",
    "그림7_SHAP_Summary_고혈압.png",
    "그림8_SHAP집단비교.png",
    "그림9_공정성지표_v2.1.png",
    "그림10_거버넌스레이더_v2.1.png",
    "그림11_Perturbation견고성.png",
    "그림12_거버넌스운영사이클.png    ← 신규: 운영 사이클 플로우차트",
]
for f in files:
    print(f"  - {f}")
print("="*60)