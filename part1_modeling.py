# ============================================================
# [Part 1] 모델링 파이프라인 — 업그레이드 v2.1
# 데이터 전처리 → Baseline(LR) 포함 알고리즘 학습 → CV
# → K-means 등급 → SHAP → 논문용 표 CSV 저장
#
# 추가/변경 사항 (v2.1):
#   - Logistic Regression Baseline 추가 (ALGOS에 'LR' 포함)
#   - 표1: 집단별 표본 수 및 유병률 (표1_집단별기술통계.csv)
#   - 표2: 알고리즘별 성능 비교 Hold-out (표2_알고리즘성능비교_HO.csv)
#   - 표3: 5-fold CV 성능 (표3_알고리즘성능비교_CV.csv)
#   - 표4: 복합건강등급 요약 (표4_복합건강등급요약.csv)
#   - 한계점 메타 정보 CSV (연구한계점_메타.csv)
#
# 출력 변수 (Part 2에서 사용):
#   df_final, all_results, best_models, best_algo_name,
#   cv_df, grade_results, shap_results,
#   X_FEATURES, DISEASES, AGEGROUP_CONFIG, GRP_NAMES, GRP_LABEL,
#   COLORS, ALGOS, SEED, DPI
# ============================================================

import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pyreadstat
import joblib
import shap
import matplotlib
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

import lightgbm as lgb
import xgboost as xgb
import optuna

# AUTO-INJECTED: Korean font setup for matplotlib
import os as _os
import matplotlib.font_manager as _fm
import matplotlib.pyplot as _plt
if not any('NanumGothic' in f.name for f in _fm.fontManager.ttflist):
    for _font in ['/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
                  '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf']:
        if _os.path.exists(_font):
            _fm.fontManager.addfont(_font)
_plt.rcParams.update({'font.family': 'NanumGothic', 'axes.unicode_minus': False})
del _os, _fm, _plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)
optuna.logging.set_verbosity(optuna.logging.WARNING)

plt.rcParams['font.family']        = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus']  = False
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

SEED = 42
DPI  = 200
np.random.seed(SEED)

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
GRP_LABEL = ["청년 남성", "청년 여성",
             "중장년 남성", "중장년 여성",
             "고령 남성",  "고령 여성"]

COLORS = ['#1565C0', '#90CAF9',
          '#E65100', '#FFCC80',
          '#2E7D32', '#A5D6A7']

DISEASES = {'Diabetes': '당뇨', 'Hypertension': '고혈압'}

# ▶ LR(Baseline) 추가
ALGOS = ['LR', 'RF', 'LGBM', 'XGB', 'MLP']


# %% =========================================================
# § 1  KNHANES 2020–2024 데이터 병합
# ============================================================

df20, _ = pyreadstat.read_sas7bdat('hn20_all.sas7bdat')
df21, _ = pyreadstat.read_sas7bdat('hn21_all.sas7bdat')
df22, _ = pyreadstat.read_sas7bdat('hn22_all.sas7bdat')
df23, _ = pyreadstat.read_sas7bdat('hn23_all.sas7bdat')
df24, _ = pyreadstat.read_sas7bdat('hn24_all.sas7bdat')

KEY_COLS = ['ID', 'year', 'sex', 'age']
CAT_COLS = [
    'HE_obe', 'BO1_1', 'BO1_2', 'BO1_3',
    'BD1_11', 'BD2_1', 'BS3_1',
    'BE3_71', 'BE3_75', 'BE3_81', 'BE3_91', 'pa_aerobic',
    'L_BR_FQ', 'BP1', 'mh_stress',
    'incm', 'ho_incm', 'edu', 'BH1',
]
NUM_COLS = [
    'HE_BMI', 'HE_wc', 'HE_wt',
    'N_EN', 'N_CHO', 'N_SUGAR', 'N_NA',
    'N_FAT', 'N_SFA', 'N_TDF', 'N_K', 'N_PROT',
]
TARGET_COLS = ['HE_DM_HbA1c', 'HE_HP']
ALL_VARS    = KEY_COLS + CAT_COLS + NUM_COLS + TARGET_COLS

df_total = pd.concat(
    [d[[v for v in ALL_VARS if v in d.columns]].copy()
     for d in [df20, df21, df22, df23, df24]],
    axis=0
).reset_index(drop=True)
print(f"병합 후 전체 행수: {len(df_total):,} / 컬럼수: {df_total.shape[1]}")

df_total['HE_DM_HbA1c'] = df_total['HE_DM_HbA1c'].map({1: 0, 3: 1})
df_total['HE_HP']        = df_total['HE_HP'].map({1: 0, 4: 1})
df_total = df_total.dropna(subset=['HE_DM_HbA1c', 'HE_HP']).reset_index(drop=True)

df_total['age'] = pd.to_numeric(df_total['age'], errors='coerce')
df_total = df_total[df_total['age'] >= 19].copy().reset_index(drop=True)

def assign_age_group(age):
    if age <= 39: return 0.0
    if age <= 59: return 1.0
    return 2.0

df_total['age_group'] = df_total['age'].apply(assign_age_group)

print("\n연령군 분포 (0=청년 19–39 / 1=중장년 40–59 / 2=고령 60+):")
print(df_total['age_group'].value_counts().sort_index())

LABEL_DICT = {
    'ID': 'ID', 'year': 'SurveyYear', 'sex': 'Sex',
    'age': 'Age', 'age_group': 'AgeGroup',
    'HE_DM_HbA1c': 'Diabetes', 'HE_HP': 'Hypertension',
    'HE_obe': 'ObesityStatus',
    'BO1_1': 'WeightChangeStatus',
    'BO1_2': 'WeightLossAmount',     'BO1_3': 'WeightGainAmount',
    'BD1_11': 'DrinkingFrequency',   'BD2_1': 'DrinkingAmount',
    'BS3_1': 'SmokingStatus',
    'BE3_71': 'VigorousActivity_Work',
    'BE3_75': 'VigorousActivity_Leisure',
    'BE3_81': 'ModerateActivity_Work',
    'BE3_91': 'WalkingActivity',
    'pa_aerobic': 'AerobicActivityRate',
    'L_BR_FQ': 'BreakfastFrequency',
    'BP1': 'StressLevel',            'mh_stress': 'StressAwarenessRate',
    'incm': 'PersonalIncomeQuartile', 'ho_incm': 'HouseholdIncomeQuartile',
    'edu': 'EducationLevel',         'BH1': 'HealthScreeningStatus',
    'HE_BMI': 'BMI',    'HE_wc': 'WaistCirc',    'HE_wt': 'Weight',
    'N_EN': 'Energy_kcal', 'N_CHO': 'Carb_g',     'N_SUGAR': 'Sugar_g',
    'N_NA': 'Sodium_mg',   'N_FAT': 'Fat_g',      'N_SFA': 'SaturatedFat_g',
    'N_TDF': 'Fiber_g',    'N_K': 'Potassium_mg', 'N_PROT': 'Protein_g',
}
df_total.rename(columns=LABEL_DICT, inplace=True)
df_total.to_csv('knhanes_preprocessed.csv', index=False, encoding='utf-8-sig')
print(">>> 전처리 완료: knhanes_preprocessed.csv")


# %% =========================================================
# § 2  피처 정의
# ============================================================

CAT_FEATURES = [
    'ObesityStatus', 'WeightChangeStatus', 'WeightLossAmount', 'WeightGainAmount',
    'DrinkingFrequency', 'DrinkingAmount', 'SmokingStatus',
    'VigorousActivity_Work', 'VigorousActivity_Leisure',
    'ModerateActivity_Work', 'WalkingActivity', 'AerobicActivityRate',
    'BreakfastFrequency', 'StressLevel', 'StressAwarenessRate',
    'PersonalIncomeQuartile', 'HouseholdIncomeQuartile',
    'EducationLevel', 'HealthScreeningStatus',
]
NUM_FEATURES = [
    'BMI', 'WaistCirc', 'Weight',
    'Energy_kcal', 'Carb_g', 'Sugar_g', 'Sodium_mg',
    'Fat_g', 'SaturatedFat_g', 'Fiber_g', 'Potassium_mg', 'Protein_g',
]
X_FEATURES = NUM_FEATURES + CAT_FEATURES

required_cols = X_FEATURES + ['Diabetes', 'Hypertension', 'Sex', 'AgeGroup', 'SurveyYear']
df_final = df_total[[c for c in required_cols if c in df_total.columns]].copy()
for col in df_final.columns:
    df_final[col] = pd.to_numeric(df_final[col], errors='coerce').fillna(0)
df_final = df_final.astype(float)

print(f"\ndf_final 형태: {df_final.shape}")


# %% =========================================================
# § 표1  집단별 기술통계 (표본 수, 유병률, 소표본 경고)
# ============================================================

print("\n[표1] 집단별 표본 수 및 유병률")
table1_rows = []
for grp, cfg in AGEGROUP_CONFIG.items():
    sub = df_final[
        (df_final['AgeGroup'] == cfg['age_group']) &
        (df_final['Sex']      == cfg['sex_code'])
    ]
    n      = len(sub)
    dm_n   = int(sub['Diabetes'].sum())
    hp_n   = int(sub['Hypertension'].sum())
    dm_r   = sub['Diabetes'].mean() * 100
    hp_r   = sub['Hypertension'].mean() * 100
    # 소표본 경고 (유병자 50명 미만)
    warn_dm = '⚠️ 소표본' if dm_n < 50 else ''
    warn_hp = '⚠️ 소표본' if hp_n < 50 else ''
    table1_rows.append({
        '집단': grp,
        '연령범위': f"{cfg['age_min']}–{cfg['age_max']}세",
        '성별': '남성' if cfg['sex_code'] == 1.0 else '여성',
        '표본수(n)': n,
        '당뇨_유병자수': dm_n,
        '당뇨_유병률(%)': round(dm_r, 1),
        '당뇨_비고': warn_dm,
        '고혈압_유병자수': hp_n,
        '고혈압_유병률(%)': round(hp_r, 1),
        '고혈압_비고': warn_hp,
        '데이터출처': 'KNHANES 2020–2024 (횡단면)',
    })

table1 = pd.DataFrame(table1_rows)
table1.to_csv('표1_집단별기술통계.csv', index=False, encoding='utf-8-sig')
print(table1.to_string(index=False))
print(">>> 저장: 표1_집단별기술통계.csv")


# %% =========================================================
# § 3  알고리즘별 학습·평가 함수 (LR Baseline 포함)
# ============================================================

def train_and_evaluate(X_tr, X_val, y_tr, y_val, algo: str, n_trials: int = 20):
    sw = compute_sample_weight('balanced', y=y_tr)

    # ── Baseline: Logistic Regression ──
    if algo == 'LR':
        model = LogisticRegression(
            max_iter=1000, random_state=SEED,
            class_weight='balanced', solver='lbfgs'
        )
        model.fit(X_tr, y_tr)

    elif algo == 'RF':
        def obj(trial):
            m = RandomForestClassifier(
                n_estimators      = trial.suggest_int('n_estimators', 100, 500),
                max_depth         = trial.suggest_int('max_depth', 3, 15),
                min_samples_split = trial.suggest_int('min_samples_split', 2, 10),
                random_state=SEED, n_jobs=-1,
            )
            m.fit(X_tr, y_tr, sample_weight=sw)
            return roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
        study = optuna.create_study(direction='maximize')
        study.optimize(obj, n_trials=n_trials)
        model = RandomForestClassifier(
            **study.best_params, random_state=SEED, n_jobs=-1)
        model.fit(X_tr, y_tr, sample_weight=sw)

    elif algo == 'LGBM':
        def obj(trial):
            m = lgb.LGBMClassifier(
                n_estimators     = trial.suggest_int('n_estimators', 100, 500),
                max_depth        = trial.suggest_int('max_depth', 3, 7),
                learning_rate    = trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                subsample        = trial.suggest_float('subsample', 0.7, 1.0),
                colsample_bytree = trial.suggest_float('colsample_bytree', 0.7, 1.0),
                random_state=SEED, n_jobs=-1, verbose=-1,
            )
            m.fit(X_tr, y_tr, sample_weight=sw)
            return roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
        study = optuna.create_study(direction='maximize')
        study.optimize(obj, n_trials=n_trials)
        model = lgb.LGBMClassifier(
            **study.best_params, random_state=SEED, n_jobs=-1, verbose=-1)
        model.fit(X_tr, y_tr, sample_weight=sw)

    elif algo == 'XGB':
        def obj(trial):
            m = xgb.XGBClassifier(
                n_estimators     = trial.suggest_int('n_estimators', 100, 500),
                max_depth        = trial.suggest_int('max_depth', 3, 7),
                learning_rate    = trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                subsample        = trial.suggest_float('subsample', 0.7, 1.0),
                colsample_bytree = trial.suggest_float('colsample_bytree', 0.7, 1.0),
                use_label_encoder=False, eval_metric='logloss',
                random_state=SEED, tree_method='hist',
            )
            m.fit(X_tr, y_tr, sample_weight=sw)
            return roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
        study = optuna.create_study(direction='maximize')
        study.optimize(obj, n_trials=n_trials)
        model = xgb.XGBClassifier(
            **study.best_params,
            use_label_encoder=False, eval_metric='logloss',
            random_state=SEED, tree_method='hist',
        )
        model.fit(X_tr, y_tr, sample_weight=sw)

    elif algo == 'MLP':
        def obj(trial):
            n_layers = trial.suggest_int('n_layers', 1, 3)
            n_units  = trial.suggest_int('n_units', 32, 256)
            m = MLPClassifier(
                hidden_layer_sizes = tuple([n_units] * n_layers),
                alpha              = trial.suggest_float('alpha', 1e-5, 1e-2, log=True),
                learning_rate_init = trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
                max_iter=300, random_state=SEED,
            )
            m.fit(X_tr, y_tr)
            return roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
        study = optuna.create_study(direction='maximize')
        study.optimize(obj, n_trials=n_trials)
        bp = study.best_params
        model = MLPClassifier(
            hidden_layer_sizes = tuple([bp['n_units']] * bp['n_layers']),
            alpha              = bp['alpha'],
            learning_rate_init = bp['learning_rate_init'],
            max_iter=300, random_state=SEED,
        )
        model.fit(X_tr, y_tr)
    else:
        raise ValueError(f"지원하지 않는 알고리즘: {algo}")

    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)

    return model, {
        'AUC':       round(roc_auc_score(y_val, y_prob), 4),
        'F1':        round(f1_score(y_val, y_pred, average='weighted'), 4),
        'Precision': round(precision_score(y_val, y_pred, average='weighted', zero_division=0), 4),
        'Recall':    round(recall_score(y_val, y_pred, average='weighted', zero_division=0), 4),
    }


# %% =========================================================
# § 4  6개 집단 × 2개 질환 × 5개 알고리즘 학습 (Hold-out)
# ============================================================

all_results    = {}
best_models    = {}
best_algo_name = {}

for grp_name, cfg in AGEGROUP_CONFIG.items():
    all_results[grp_name]    = {}
    best_models[grp_name]    = {}
    best_algo_name[grp_name] = {}

    df_g = df_final[
        (df_final['AgeGroup'] == cfg['age_group']) &
        (df_final['Sex']      == cfg['sex_code'])
    ].copy()
    print(f"\n{'='*20} [{grp_name}] 표본 수: {len(df_g):,}명 {'='*20}")

    for dis_col, dis_name in DISEASES.items():
        print(f"\n  ── [{dis_name}] 알고리즘 비교 ──")
        all_results[grp_name][dis_col] = {}

        X = df_g[X_FEATURES]
        y = df_g[dis_col].astype(int)

        if y.sum() < 20:
            print(f"    ⚠️  유병자 {int(y.sum())}명 — 모델 학습 생략")
            best_models[grp_name][dis_col]    = None
            best_algo_name[grp_name][dis_col] = None
            continue

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.3, random_state=SEED, stratify=y)

        best_auc, best_model, best_algo = -1, None, None

        for algo in ALGOS:
            try:
                model, metrics = train_and_evaluate(X_tr, X_val, y_tr, y_val, algo)
                all_results[grp_name][dis_col][algo] = metrics
                print(f"    {algo:6s} | AUC={metrics['AUC']:.4f} "
                      f"F1={metrics['F1']:.4f} "
                      f"Prec={metrics['Precision']:.4f} "
                      f"Rec={metrics['Recall']:.4f}"
                      + (" ← Baseline" if algo == 'LR' else ""))
                # LR은 baseline 참조용 — 최적 모델 선정에서 제외
                if algo != 'LR' and metrics['AUC'] > best_auc:
                    best_auc, best_model, best_algo = metrics['AUC'], model, algo
            except Exception as e:
                print(f"    {algo:6s} | 오류: {e}")
                all_results[grp_name][dis_col][algo] = None

        best_models[grp_name][dis_col]    = best_model
        best_algo_name[grp_name][dis_col] = best_algo
        print(f"  → 최적(앙상블): {best_algo} (AUC={best_auc:.4f})")


# %% =========================================================
# § 표2  알고리즘 성능 비교 (Hold-out) — LR Baseline 포함
# ============================================================

rows = []
for grp, dis_dict in all_results.items():
    for dis, algo_dict in dis_dict.items():
        for algo, metrics in algo_dict.items():
            if metrics:
                is_best = (algo == best_algo_name.get(grp, {}).get(dis))
                is_baseline = (algo == 'LR')
                rows.append({
                    '집단': grp,
                    '질환': DISEASES.get(dis, dis),
                    '알고리즘': algo,
                    '구분': 'Baseline' if is_baseline else ('최적' if is_best else '비교'),
                    'AUC':       metrics['AUC'],
                    'F1':        metrics['F1'],
                    'Precision': metrics['Precision'],
                    'Recall':    metrics['Recall'],
                    'Baseline_대비_AUC향상': round(
                        metrics['AUC'] - (algo_dict.get('LR') or {}).get('AUC', metrics['AUC']), 4
                    ) if not is_baseline and algo_dict.get('LR') else '-',
                })

table2 = pd.DataFrame(rows)
table2.to_csv('표2_알고리즘성능비교_HO.csv', index=False, encoding='utf-8-sig')
print("\n>>> 저장: 표2_알고리즘성능비교_HO.csv")

# 성능비교_holdout.csv도 유지 (기존 호환)
table2.to_csv('성능비교_holdout.csv', index=False, encoding='utf-8-sig')


# %% =========================================================
# § 4-1  5-fold 교차검증
# ============================================================

params_dict = {}
for grp_name in AGEGROUP_CONFIG:
    params_dict[grp_name] = {}
    for dis_col in DISEASES:
        mdl = best_models.get(grp_name, {}).get(dis_col)
        if mdl is None:
            params_dict[grp_name][dis_col] = {}
        elif hasattr(mdl, 'get_params'):
            params_dict[grp_name][dis_col] = mdl.get_params()
        else:
            params_dict[grp_name][dis_col] = {}

def run_cv(df_grp, algo, disease, grp_name, n_splits=5):
    X  = df_grp[X_FEATURES]
    y  = df_grp[disease].astype(int)
    bp = params_dict.get(grp_name, {}).get(disease, {})
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    fold_auc, fold_f1, fold_prec, fold_rec = [], [], [], []

    for tr_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        sw = compute_sample_weight('balanced', y=y_tr)

        if algo == 'LR':
            mdl = LogisticRegression(
                max_iter=1000, random_state=SEED,
                class_weight='balanced', solver='lbfgs')
            mdl.fit(X_tr, y_tr)
        elif algo == 'RF':
            mdl = RandomForestClassifier(
                **{k: v for k, v in bp.items()
                   if k in ['n_estimators', 'max_depth', 'min_samples_split']},
                random_state=SEED, n_jobs=-1)
            mdl.fit(X_tr, y_tr, sample_weight=sw)
        elif algo == 'LGBM':
            mdl = lgb.LGBMClassifier(
                **{k: v for k, v in bp.items()
                   if k in ['n_estimators', 'max_depth', 'learning_rate',
                             'subsample', 'colsample_bytree']},
                random_state=SEED, n_jobs=-1, verbose=-1)
            mdl.fit(X_tr, y_tr, sample_weight=sw)
        elif algo == 'XGB':
            mdl = xgb.XGBClassifier(
                **{k: v for k, v in bp.items()
                   if k in ['n_estimators', 'max_depth', 'learning_rate',
                             'subsample', 'colsample_bytree']},
                use_label_encoder=False, eval_metric='logloss',
                random_state=SEED, tree_method='hist')
            mdl.fit(X_tr, y_tr, sample_weight=sw)
        elif algo == 'MLP':
            mdl = MLPClassifier(
                hidden_layer_sizes=tuple(
                    [bp.get('n_units', 128)] * bp.get('n_layers', 2)),
                alpha=bp.get('alpha', 1e-4),
                learning_rate_init=bp.get('learning_rate_init', 1e-3),
                max_iter=300, random_state=SEED)
            mdl.fit(X_tr, y_tr)
        else:
            continue

        y_prob = mdl.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        fold_auc.append(roc_auc_score(y_val, y_prob))
        fold_f1.append(f1_score(y_val, y_pred, average='weighted', zero_division=0))
        fold_prec.append(precision_score(y_val, y_pred, average='weighted', zero_division=0))
        fold_rec.append(recall_score(y_val, y_pred, average='weighted', zero_division=0))

    def fmt(arr): return f"{np.mean(arr):.4f} ± {np.std(arr):.4f}"
    return {
        'CV_AUC': fmt(fold_auc), 'CV_F1': fmt(fold_f1),
        'CV_Precision': fmt(fold_prec), 'CV_Recall': fmt(fold_rec),
        '_raw_auc': fold_auc,
    }

print("\n" + "="*60)
print("5-fold 교차검증")
print("="*60)

cv_rows = []
for grp_name, cfg in AGEGROUP_CONFIG.items():
    df_g = df_final[
        (df_final['AgeGroup'] == cfg['age_group']) &
        (df_final['Sex']      == cfg['sex_code'])
    ].copy()

    for dis_col, dis_name in DISEASES.items():
        algo = best_algo_name.get(grp_name, {}).get(dis_col)
        if algo is None:
            continue
        print(f"\n  [{grp_name}] [{dis_name}] — {algo} CV 실행 중...")
        cv_result = run_cv(df_g, algo, dis_col, grp_name)
        ho = all_results[grp_name][dis_col].get(algo, {})

        # LR baseline CV도 함께 수행
        lr_cv = run_cv(df_g, 'LR', dis_col, grp_name)
        lr_ho = all_results[grp_name][dis_col].get('LR', {})

        cv_rows.append({
            '집단': grp_name, '질환': dis_name, '최적알고리즘': algo,
            'HO_AUC': ho.get('AUC', '-'), 'HO_F1': ho.get('F1', '-'),
            'HO_Precision': ho.get('Precision', '-'),
            'HO_Recall': ho.get('Recall', '-'),
            **{k: v for k, v in cv_result.items() if not k.startswith('_')},
            'LR_HO_AUC': lr_ho.get('AUC', '-'),
            'LR_CV_AUC': lr_cv['CV_AUC'],
            'AUC_vs_Baseline': round(
                ho.get('AUC', 0) - lr_ho.get('AUC', 0), 4
            ) if lr_ho.get('AUC') else '-',
        })
        print(f"    Hold-out AUC : {ho.get('AUC', '-')}")
        print(f"    CV AUC       : {cv_result['CV_AUC']}")
        print(f"    LR Baseline  : {lr_ho.get('AUC', '-')} (HO) / {lr_cv['CV_AUC']} (CV)")

cv_df = pd.DataFrame(cv_rows)

# § 표3 저장
table3 = cv_df.copy()
table3.to_csv('표3_알고리즘성능비교_CV.csv', index=False, encoding='utf-8-sig')
cv_df.to_csv('성능비교_표2_최종.csv', index=False, encoding='utf-8-sig')
print("\n>>> 저장: 표3_알고리즘성능비교_CV.csv / 성능비교_표2_최종.csv")


# %% =========================================================
# § 5  K-means 복합건강등급 산정
# ============================================================

def assign_kmeans_grade(prob_dm, prob_hp, n_clusters=3, random_state=SEED):
    X_km       = np.column_stack([prob_dm, prob_hp])
    km         = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels     = km.fit_predict(X_km)
    rank_order = np.argsort(km.cluster_centers_.sum(axis=1))
    grade_map  = {old: new + 1 for new, old in enumerate(rank_order)}
    return np.vectorize(grade_map.get)(labels), km

grade_results = {}
table4_rows   = []

for grp_name, cfg in AGEGROUP_CONFIG.items():
    print(f"\n{'='*20} [{grp_name}] 복합건강등급 {'='*20}")
    df_g = df_final[
        (df_final['AgeGroup'] == cfg['age_group']) &
        (df_final['Sex']      == cfg['sex_code'])
    ].copy().reset_index(drop=True)

    mdl_dm = best_models[grp_name].get('Diabetes')
    mdl_hp = best_models[grp_name].get('Hypertension')
    if mdl_dm is None or mdl_hp is None:
        print("  ⚠️  모델 없음 — 건너뜀")
        continue

    X       = df_g[X_FEATURES]
    prob_dm = mdl_dm.predict_proba(X)[:, 1]
    prob_hp = mdl_hp.predict_proba(X)[:, 1]

    grades, _ = assign_kmeans_grade(prob_dm, prob_hp)
    df_g['prob_DM']      = prob_dm
    df_g['prob_HP']      = prob_hp
    df_g['복합건강등급'] = grades

    summary = df_g.groupby('복합건강등급').agg(
        건수           = ('복합건강등급', 'count'),
        당뇨유병률     = ('Diabetes',     'mean'),
        고혈압유병률   = ('Hypertension', 'mean'),
        평균당뇨확률   = ('prob_DM',      'mean'),
        평균고혈압확률 = ('prob_HP',      'mean'),
    ).round(4)
    summary['구성비'] = (summary['건수'] / summary['건수'].sum()).round(4)
    print(summary.to_string())

    # § 표4 누적
    grade_label = {1: '1등급(저위험)', 2: '2등급(중위험)', 3: '3등급(고위험)'}
    # 정책 해석 추가
    policy_map = {
        1: '예방적 건강관리 서비스 연계 권고',
        2: '주기적 모니터링 + 생활습관 개입',
        3: '의료기관 연계 및 집중 관리',
    }
    for g in [1, 2, 3]:
        if g not in summary.index:
            continue
        row = summary.loc[g]
        table4_rows.append({
            '집단': grp_name,
            '등급': grade_label[g],
            '건수': int(row['건수']),
            '구성비(%)': round(row['구성비'] * 100, 1),
            '당뇨유병률(%)': round(row['당뇨유병률'] * 100, 1),
            '고혈압유병률(%)': round(row['고혈압유병률'] * 100, 1),
            '평균당뇨예측확률': row['평균당뇨확률'],
            '평균고혈압예측확률': row['평균고혈압확률'],
            '정책적함의': policy_map[g],
        })

    grade_results[grp_name] = df_g.copy()
    df_g.to_csv(f'등급결과_{grp_name}.csv', index=False, encoding='utf-8-sig')

# § 표4 저장
table4 = pd.DataFrame(table4_rows)
table4.to_csv('표4_복합건강등급요약.csv', index=False, encoding='utf-8-sig')
print("\n>>> 저장: 표4_복합건강등급요약.csv")


# %% =========================================================
# § 6  SHAP 분석
# ============================================================

def run_shap_analysis(model, X_val, algo, grp_name, dis_name, n_sample=500):
    print(f"\n  SHAP: [{grp_name}] [{dis_name}] ({algo})")
    X_s = X_val.sample(min(n_sample, len(X_val)), random_state=SEED)

    if algo in ('RF', 'LGBM', 'XGB'):
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_s)
        if isinstance(shap_values, list):
            sv = shap_values[1]
        elif shap_values.ndim == 3:
            sv = shap_values[:, :, 1]
        else:
            sv = shap_values
    else:
        X_bg      = shap.sample(X_s, min(100, len(X_s)))
        explainer = shap.KernelExplainer(
            lambda x: model.predict_proba(x)[:, 1], X_bg)
        sv  = explainer.shap_values(X_s.iloc[:50])
        X_s = X_s.iloc[:50]

    plt.figure(figsize=(8, 6))
    shap.summary_plot(sv, X_s, plot_type='dot', show=False, max_display=15)
    plt.tight_layout()
    fname = f'shap_summary_{grp_name}_{dis_name}.png'
    plt.savefig(fname, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"    저장: {fname}")

    probs      = model.predict_proba(X_s)[:, 1]
    border_idx = np.where((probs >= 0.35) & (probs <= 0.65))[0]
    idx        = border_idx[0] if len(border_idx) > 0 else 0
    ev         = explainer.expected_value
    base_val   = float(ev[1] if isinstance(ev, (list, np.ndarray)) else ev)

    plt.figure(figsize=(8, 5))
    shap.waterfall_plot(
        shap.Explanation(
            values=sv[idx], base_values=base_val,
            data=X_s.iloc[idx].values,
            feature_names=X_s.columns.tolist(),
        ), show=False, max_display=12,
    )
    plt.tight_layout()
    fname2 = f'shap_individual_{grp_name}_{dis_name}.png'
    plt.savefig(fname2, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"    저장: {fname2}")

    return sv, X_s

shap_results = {}
for grp_name, cfg in AGEGROUP_CONFIG.items():
    shap_results[grp_name] = {}
    df_g = df_final[
        (df_final['AgeGroup'] == cfg['age_group']) &
        (df_final['Sex']      == cfg['sex_code'])
    ].copy()
    X = df_g[X_FEATURES]
    print(f"\n{'='*20} [{grp_name}] SHAP {'='*20}")

    for dis_col, dis_name in DISEASES.items():
        mdl  = best_models[grp_name].get(dis_col)
        algo = best_algo_name[grp_name].get(dis_col)
        if mdl is None:
            continue
        y = df_g[dis_col].astype(int)
        _, X_val, _, _ = train_test_split(
            X, y, test_size=0.3, random_state=SEED, stratify=y)
        sv, X_s = run_shap_analysis(mdl, X_val, algo, grp_name, dis_name)
        shap_results[grp_name][dis_col] = {'shap_values': sv, 'X_sample': X_s}


# %% =========================================================
# § 연구 한계점 메타 정보 CSV
# ============================================================

limitations = pd.DataFrame([
    {
        '번호': 1,
        '한계유형': '데이터 구조',
        '내용': 'KNHANES는 연도별 독립 횡단면으로 동일 개인 종단 추적 불가',
        '영향': 'Temporal Validation을 단일 연도(2024) 분할로 제한',
        '대응': '논문 본문 명시 + 향후 코호트 DB 연계 연구 제안',
        '향후연구': '국민건강보험공단 코호트 DB 연계 시 종단 Temporal Validation 가능',
    },
    {
        '번호': 2,
        '한계유형': '소표본 집단',
        '내용': '청년층(19–39세) 유병자 수 부족으로 모델 불안정 가능',
        '영향': 'AUC 신뢰구간 넓음, 결과 해석 주의 필요',
        '대응': '표1 소표본 경고 명시, 결과 절에서 해석 주의 문구 삽입',
        '향후연구': '청년층 대상 대규모 코호트 확보 필요',
    },
    {
        '번호': 3,
        '한계유형': '거버넌스 운영',
        '내용': 'Human-in-the-loop 및 개인정보영향평가(PIA) 미충족',
        '영향': 'Model Card Score 감점 (8/10)',
        '대응': '한계점 절 명시 + 운영 프레임워크 제안에서 필요성 기술',
        '향후연구': '실제 건강검진 기관 적용 시 IRB·PIA 절차 설계 필요',
    },
    {
        '번호': 4,
        '한계유형': '임계값 도메인 적합성',
        '내용': '거버넌스 임계값 일부가 금융 AI 기준을 의료 도메인에 차용',
        '영향': '6집단 층화 구조로 완화 적용하였으나 의료 전용 기준 부재',
        '대응': '논문 본문에서 완화 적용 근거 명시',
        '향후연구': '의료 AI 거버넌스 전용 임계값 체계 수립 연구 필요',
    },
    {
        '번호': 5,
        '한계유형': '인과성',
        '내용': '횡단면 이진분류 기반으로 인과관계 추론 불가',
        '영향': '예측 모델로서의 활용에 한정, 인과 해석 주의',
        '대응': '연구 목적을 거버넌스 프레임워크 검증 테스트베드로 명확화',
        '향후연구': '종단 코호트 기반 인과 모델 확장',
    },
])
limitations.to_csv('연구한계점_메타.csv', index=False, encoding='utf-8-sig')
print("\n>>> 저장: 연구한계점_메타.csv")


# %% =========================================================
# § 모델 저장
# ============================================================

for grp_name in AGEGROUP_CONFIG:
    for dis_col in DISEASES:
        mdl = best_models.get(grp_name, {}).get(dis_col)
        if mdl is not None:
            joblib.dump(mdl, f'model_{grp_name}_{dis_col}.pkl')

pd.concat(
    [df_g.assign(그룹=grp) for grp, df_g in grade_results.items()],
    ignore_index=True
).to_csv('복합건강등급_전체.csv', index=False, encoding='utf-8-sig')

print("\n" + "="*55)
print(">>> Part 1 완료 — 생성 파일 목록")
print("="*55)
for f in [
    'knhanes_preprocessed.csv',
    '표1_집단별기술통계.csv',
    '표2_알고리즘성능비교_HO.csv',
    '표3_알고리즘성능비교_CV.csv',
    '표4_복합건강등급요약.csv',
    '연구한계점_메타.csv',
    '복합건강등급_전체.csv',
]:
    print(f"  - {f}")
print("="*55)

# ── Part 2를 위한 변수 직렬화 저장 ──

import joblib, pickle

df_final.to_parquet('df_final.parquet', index=False)
joblib.dump(all_results,    'all_results.pkl')
joblib.dump(best_models,    'best_models.pkl')
joblib.dump(best_algo_name, 'best_algo_name.pkl')
joblib.dump(shap_results,   'shap_results.pkl')
joblib.dump(grade_results,  'grade_results.pkl')
joblib.dump(cv_df,          'cv_df.pkl')
print(">>> Part 2용 변수 직렬화 저장 완료")