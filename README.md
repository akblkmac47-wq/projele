
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline

from collections import Counter

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (6, 4)


data_path = r"/Users/yutaev/Downloads/archive"  # kendi yoluna göre değiştir

train_file = os.path.join(data_path, "cell2celltrain.csv")
holdout_file = os.path.join(data_path, "cell2cellholdout.csv")

df_train = pd.read_csv(train_file)
df_holdout = pd.read_csv(holdout_file)

print("Train shape:", df_train.shape)
print("Holdout shape:", df_holdout.shape)

import pandas as pd
import os

data_path = r"/Users/yutaev/Downloads/archive"  # kendi yolun
train_file = os.path.join(data_path, "cell2celltrain.csv")
holdout_file = os.path.join(data_path, "cell2cellholdout.csv")

df_train = pd.read_csv(train_file)
df_holdout = pd.read_csv(holdout_file)

print(df_train.shape, df_holdout.shape)

# Hücre 2
# ============================
# İLK SATIRLAR
# ============================
df_train.head()

# Hücre 3
# ============================
# GENEL BİLGİ (info)
# ============================
df_train.info()

# Hücre 4
# ============================
# SAYISAL VE KATEGORİK SÜTUNLAR
# ============================
numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df_train.select_dtypes(exclude=[np.number]).columns.tolist()

print("Sayısal sütun sayısı:", len(numeric_cols))
print("Kategorik sütun sayısı:", len(categorical_cols))

print("\n🔸 Sayısal sütun örnekleri:", numeric_cols[:10])
print("🔹 Kategorik sütun örnekleri:", categorical_cols[:10])

# Hücre 5
# ============================
# ÖZET İSTATİSTİKLER
# ============================
df_train.describe().T

# Hücre 6
# ============================
# EKSİK (NaN) DEĞER ANALİZİ
# ============================
missing_counts = df_train.isnull().sum()
total_rows = len(df_train)

missing_percentage = (missing_counts[missing_counts > 0] / total_rows) * 100

missing_data_summary = pd.DataFrame({
    "Kayıp Sayısı": missing_counts[missing_counts > 0],
    "Kayıp Yüzdesi (%)": missing_percentage.round(2)
}).sort_values("Kayıp Sayısı", ascending=False)

missing_data_summary

# Hücre 7

plt.figure(figsize=(5, 4))
sns.countplot(x='Churn', data=df_train, order=['No', 'Yes'])
plt.xlabel('Churn')
plt.ylabel('Adet')
plt.title('Churn Dağılımı')
plt.show()

num_cols_critical = [
    'MonthlyRevenue',
    'TotalRecurringCharge',
    'MonthlyMinutes',
    'OverageMinutes',
    'DirectorAssistedCalls'
]

for col in num_cols_critical:
    if col in df_train.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x='Churn', y=col, data=df_train, order=['No', 'Yes'])
        plt.title(f'{col} Dağılımı vs Churn')
        plt.xlabel('Churn')
        plt.ylabel(col)
        plt.show()
    else:
        print(f"Uyarı: {col} sütunu df_train içinde bulunamadı.")

df_train_enc = df_train.copy()
df_holdout_enc = df_holdout.copy()

df_train_enc["Churn"] = df_train_enc["Churn"].map({"Yes": 1, "No": 0})
df_holdout_enc["Churn"] = df_holdout_enc["Churn"].map({"Yes": 1, "No": 0})

y_train = df_train_enc["Churn"]
y_holdout = df_holdout_enc["Churn"]

print("Churn sınıf dağılımı (train):")
print(y_train.value_counts(normalize=True).round(3))


if "IncomeGroup" in df_train_enc.columns:
    income_churn = df_train_enc.groupby('IncomeGroup')["Churn"].mean().sort_values()
    plt.figure(figsize=(8, 4))
    income_churn.plot(kind='bar', color='skyblue')
    plt.ylabel('Ortalama Churn Oranı')
    plt.title('IncomeGroup Bazlı Ortalama Churn Oranı')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("IncomeGroup sütunu bulunamadı.")


if "ServiceArea" in df_train_enc.columns:
    service_area_mode = df_train_enc["ServiceArea"].mode()[0]
    df_train_enc["ServiceArea"] = df_train_enc["ServiceArea"].fillna(service_area_mode)
    df_holdout_enc["ServiceArea"] = df_holdout_enc["ServiceArea"].fillna(service_area_mode)

# Sayısal sütunları train ortalaması ile doldur
numeric_cols_enc = df_train_enc.select_dtypes(include=[np.number]).columns.tolist()

for col in numeric_cols_enc:
    mean_val = df_train_enc[col].mean()
    df_train_enc[col] = df_train_enc[col].fillna(mean_val)
    df_holdout_enc[col] = df_holdout_enc[col].fillna(mean_val)

print("Eksik değer doldurma (encoded) tamamlandı.")



categorical_cols_enc = df_train_enc.select_dtypes(exclude=[np.number]).columns.tolist()
categorical_cols_enc = [c for c in categorical_cols_enc if c != "Churn"]

for col in categorical_cols_enc:
    le = LabelEncoder()
    combined = pd.concat([df_train_enc[col], df_holdout_enc[col]], axis=0).astype(str)
    le.fit(combined)

    df_train_enc[col] = le.transform(df_train_enc[col].astype(str))
    df_holdout_enc[col] = le.transform(df_holdout_enc[col].astype(str))

print("LabelEncoder işlemi tamamlandı.")

drop_cols = ["Churn"]
if "CustomerID" in df_train_enc.columns:
    drop_cols.append("CustomerID")

X_train = df_train_enc.drop(columns=drop_cols)
X_holdout = df_holdout_enc.drop(columns=drop_cols)

y_train = df_train_enc["Churn"]
y_holdout = df_holdout_enc["Churn"]

print("X_train shape:", X_train.shape)
print("X_holdout shape:", X_holdout.shape)


scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = pd.DataFrame(
    scaler.transform(X_train),
    columns=X_train.columns
)
X_holdout_scaled = pd.DataFrame(
    scaler.transform(X_holdout),
    columns=X_holdout.columns
)

print("Ölçeklendirme tamamlandı.")



# Kişi 1: SelectKBest Sonuçları
kbest_features = [
    'CustomerID', 'MonthlyMinutes', 'TotalRecurringCharge', 'PercChangeMinutes', 
    'PercChangeRevenues', 'BlockedCalls', 'InboundCalls', 'PeakCallsInOut', 'OffPeakCallsInOut',
    'MonthsInService', 'UniqueSubs', 'ActiveSubs', 'ServiceArea', 'CurrentEquipmentDays', 
    'HandsetWebCapable', 'Homeownership', 'BuysViaMailOrder', 'RespondsToMailOffers', 'HasCreditCard',
    'RetentionCalls', 'AdjustmentsToCreditRating', 'HandsetPrice', 'CreditRating', 'Occupation', 'MaritalStatus'
]

# Kişi 2: Random Forest Importance Sonuçları
rf_features = [
    'CurrentEquipmentDays','CustomerID','PercChangeMinutes','MonthlyMinutes','MonthsInService',
    'MonthlyRevenue','PercChangeRevenues','ServiceArea','PeakCallsInOut','OffPeakCallsInOut',
    'ReceivedCalls','UnansweredCalls','OutboundCalls','DroppedBlockedCalls',
    'TotalRecurringCharge','DroppedCalls','AgeHH1','InboundCalls','OverageMinutes','BlockedCalls'
]

# Kişi 3: RFE (Lojistik Regresyon) Sonuçları
rfe_features = [
    'MonthlyRevenue','MonthlyMinutes','TotalRecurringCharge','PercChangeMinutes','PercChangeRevenues','DroppedCalls','BlockedCalls',
    'DroppedBlockedCalls','MonthsInService','UniqueSubs','ActiveSubs','CurrentEquipmentDays','AgeHH1','HandsetRefurbished','MadeCallToRetentionTeam'
]

# Kişi 4: Lasso (L1 Regülasyonu) Sonuçları
lasso_features = [
    'MonthlyRevenue','MonthlyMinutes','TotalRecurringCharge','OverageMinutes','PercChangeMinutes','PercChangeRevenues','DroppedCalls','BlockedCalls',
    'CustomerCareCalls','ThreewayCalls','OutboundCalls','PeakCallsInOut','DroppedBlockedCalls','MonthsInService','UniqueSubs','ActiveSubs','Handsets',
    'CurrentEquipmentDays','AgeHH1','ChildrenInHH','HandsetRefurbished','RespondsToMailOffers','HasCreditCard','IncomeGroup','AdjustmentsToCreditRating',
    'HandsetPrice','MadeCallToRetentionTeam','CreditRating'
]

print("Tüm listeler hafızaya alındı. Toplam listeler:")
print(f"KBest Listesi: {len(kbest_features)} özellik")
print(f"Random Forest Listesi: {len(rf_features)} özellik")
print(f"RFE Listesi: {len(rfe_features)} özellik")
print(f"Lasso Listesi: {len(lasso_features)} özellik")


from collections import Counter

all_lists = {
    "kbest": kbest_features,
    "rf": rf_features,
    "rfe": rfe_features,
    "lasso": lasso_features
}

# Hangi özellik kaç listede geçmiş?
counter = Counter()
for name, feat_list in all_lists.items():
    counter.update(feat_list)

feature_vote_df = pd.DataFrame.from_dict(counter, orient='index', columns=['Listelerde Geçme Sayısı'])
feature_vote_df = feature_vote_df.sort_values('Listelerde Geçme Sayısı', ascending=False)

print("Özelliklerin listelerde görünme sayıları (ilk 20):")
display(feature_vote_df.head(20))

print("\nLASSO listesinde toplam özellik sayısı:", len(lasso_features))

# Ölçeklenmiş X_train içinde gerçekten olan özellikleri filtreleyelim
final_lasso_features = [f for f in lasso_features if f in X_train_scaled.columns]

print("X_train_scaled içinde bulunup kullanılacak LASSO özellik sayısı:", len(final_lasso_features))
print("Kullanılacak özellikler:")
print(final_lasso_features)

# Bu 28 özellikten yeni X matrisi oluştur
X_train_fs = X_train_scaled[final_lasso_features].copy()
X_holdout_fs = X_holdout_scaled[final_lasso_features].copy()

print("\nX_train_fs shape:", X_train_fs.shape)
print("X_holdout_fs shape:", X_holdout_fs.shape)

for v in ["X_train_fs","X_holdout_fs","y_train","y_holdout"]:
    assert v in globals(), f"{v} eksik; hücre 9–16'yı çalıştır."

# Hücre 17
# ============================
# MODEL 1: LassoCV (L1 cezalı lineer model, 28 LASSO özelliğiyle)
# ============================
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

X_for_lasso = X_train_fs.values
y_for_lasso = y_train.values

X_tr_lasso, X_te_lasso, y_tr_lasso, y_te_lasso = train_test_split(
    X_for_lasso, y_for_lasso,
    test_size=0.2,
    random_state=42,
    stratify=y_for_lasso
)

lasso_model = LassoCV(
    alphas=None,
    cv=5,
    random_state=42
)

lasso_model.fit(X_tr_lasso, y_tr_lasso)

print("Seçilen alpha (LassoCV):", lasso_model.alpha_)

# Sürekli tahminler
y_pred_lasso_cont = lasso_model.predict(X_te_lasso)
y_pred_lasso_cls = (y_pred_lasso_cont >= 0.5).astype(int)

print("\nModel 1 - LassoCV (Regression + 0.5 threshold)")
print("Accuracy (test):", accuracy_score(y_te_lasso, y_pred_lasso_cls))
print("F1 (test):", f1_score(y_te_lasso, y_pred_lasso_cls))
print("\nClassification Report (test):\n", classification_report(y_te_lasso, y_pred_lasso_cls))
print("Confusion Matrix (test):\n", confusion_matrix(y_te_lasso, y_pred_lasso_cls))

# Holdout performansı
# y_true'yu direkt y_holdout'tan alalım, karışıklık olmasın
y_hold_true = y_holdout.values.astype(int)

# Holdout performansı
y_hold_true = y_holdout.values.astype(int)

y_hold_pred_lasso_cont = lasso_model.predict(X_holdout_fs.values)
y_hold_pred_lasso_cls = (y_hold_pred_lasso_cont >= 0.5).astype(int)

print("\nHoldout Accuracy (LassoCV):", accuracy_score(y_hold_true, y_hold_pred_lasso_cls))
print("Holdout F1 (LassoCV):", f1_score(y_hold_true, y_hold_pred_lasso_cls))

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LassoCV

# 1) Hedef ve X'i ayır
target_col = 'Churn' if 'Churn' in df_train.columns else 'ChurnFlag'

# Yes/No ise 1/0'a çevir
if df_train[target_col].dtype == 'object':
    y = df_train[target_col].map({'Yes': 1, 'No': 0}).astype(int)
else:
    y = df_train[target_col].astype(int)

drop_cols = [c for c in ['CustomerID', target_col] if c in df_train.columns]
X = df_train.drop(columns=drop_cols)

# 2) Sayısal / kategorik kolonları bul
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

# 3) Ön işleme (imputer + scaler + OHE)
pre = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ]), num_cols),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), cat_cols)
], remainder="drop")

# 4) Train / Valid böl
X_tr, X_va, y_tr, y_va = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Train:", X_tr.shape, "| Valid:", X_va.shape)

# 5) Ön işlemi sadece train'e fit et, sonra dönüştür
pre.fit(X_tr)
X_tr_scaled = pre.transform(X_tr)
X_va_scaled = pre.transform(X_va)

feature_names = pre.get_feature_names_out()
print("Dönüştürülmüş feature sayısı:", len(feature_names))

# 6) LASSO ile boyut indirgeme (feature selection)
lasso_fs = LassoCV(cv=5, random_state=42)
lasso_fs.fit(X_tr_scaled, y_tr)

coef_abs = np.abs(lasso_fs.coef_)
sorted_idx = np.argsort(coef_abs)[::-1]

# Kaç özellik bırakmak istiyorsun? (ör: 28)
n_features = 28
top_idx = sorted_idx[:n_features]

selected_features = feature_names[top_idx]

print(f"\nSeçilen feature sayısı: {len(selected_features)}")
print("Seçilen feature'lar:")
for f in selected_features:
    print(" -", f)

# 7) Boyutu indirgenmiş X'ler
X_tr_red = X_tr_scaled[:, top_idx]
X_va_red = X_va_scaled[:, top_idx]

print("\nX_tr_red shape:", X_tr_red.shape)
print("X_va_red shape:", X_va_red.shape)

from sklearn.linear_model import LogisticRegression

pipe_l1 = Pipeline([
    ("prep", pre),
    ("clf", LogisticRegression(
        penalty="l1",
        solver="liblinear",
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ))
])

pipe_l1.fit(X_tr, y_tr)

proba_l1 = pipe_l1.predict_proba(X_va)[:, 1]
pred_l1  = (proba_l1 >= 0.5).astype(int)

def report_metrics(name, y_true, prob, thr=0.5):
    yhat = (prob >= thr).astype(int)
    return pd.Series({
        "ROC-AUC": roc_auc_score(y_true, prob),
        "PR-AUC": average_precision_score(y_true, prob),
        "B-Acc": balanced_accuracy_score(y_true, yhat),
        "F1": f1_score(y_true, yhat),
        "Acc": accuracy_score(y_true, yhat)
    }, name=name)

m_l1 = report_metrics("L1-LogReg (valid)", y_va, proba_l1)
display(m_l1)

from sklearn.ensemble import GradientBoostingClassifier

pipe_gb = Pipeline([
    ("prep", pre),
    ("clf", GradientBoostingClassifier(random_state=42))
])

pipe_gb.fit(X_tr, y_tr)

proba_gb = pipe_gb.predict_proba(X_va)[:, 1]
pred_gb  = (proba_gb >= 0.5).astype(int)

m_gb = report_metrics("GB (valid)", y_va, proba_gb)
display(m_gb)

fpr1, tpr1, _ = roc_curve(y_va, proba_l1)
fpr2, tpr2, _ = roc_curve(y_va, proba_gb)
prec1, rec1, _ = precision_recall_curve(y_va, proba_l1)
prec2, rec2, _ = precision_recall_curve(y_va, proba_gb)

plt.figure(figsize=(6, 5))
plt.plot(fpr1, tpr1, label=f"L1-LogReg ROC={roc_auc_score(y_va, proba_l1):.3f}")
plt.plot(fpr2, tpr2, label=f"GB ROC={roc_auc_score(y_va, proba_gb):.3f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve"); plt.legend(); plt.show()

plt.figure(figsize=(6, 5))
plt.plot(rec1, prec1, label=f"L1-LogReg PR={average_precision_score(y_va, proba_l1):.3f}")
plt.plot(rec2, prec2, label=f"GB PR={average_precision_score(y_va, proba_gb):.3f}")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve"); plt.legend(); plt.show()

from sklearn.metrics import ConfusionMatrixDisplay
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ConfusionMatrixDisplay.from_predictions(y_va, pred_l1, normalize='true', ax=axes[0], cmap="Blues")
axes[0].set_title("L1-LogReg")
ConfusionMatrixDisplay.from_predictions(y_va, pred_gb, normalize='true', ax=axes[1], cmap="Greens")
axes[1].set_title("GB")
plt.tight_layout()
plt.show()

# Holdout değerlendirme / inference
X_hold_full = df_holdout.drop(columns=drop_cols, errors='ignore')

# Churn etiketi varsa ve en az bir satır doluysa
if target_col in df_holdout and df_holdout[target_col].notna().any():
    mask = df_holdout[target_col].notna()
    X_hold = df_holdout.loc[mask].drop(columns=drop_cols, errors='ignore')
    y_hold = df_holdout.loc[mask, target_col].astype(int)

    hold_proba_l1 = pipe_l1.predict_proba(X_hold)[:, 1]
    hold_proba_gb = pipe_gb.predict_proba(X_hold)[:, 1]

    metrics_table = pd.DataFrame([
        report_metrics("L1-LogReg (holdout)", y_hold, hold_proba_l1),
        report_metrics("GB (holdout)", y_hold, hold_proba_gb)
    ])
    display(metrics_table)
    print(f"Holdout’ta kullanılan satır sayısı: {len(y_hold)}")
else:
    # Etiket yoksa sadece tahmin üret
    hold_proba_l1 = pipe_l1.predict_proba(X_hold_full)[:, 1]
    hold_proba_gb = pipe_gb.predict_proba(X_hold_full)[:, 1]
    preds = pd.DataFrame({
        "L1_prob": hold_proba_l1,
        "GB_prob": hold_proba_gb
    })
    print("Holdout’ta etiket bulunamadı; sadece olasılık üretildi. İlk satırlar:")
    display(preds.head())

preds = pd.DataFrame({
    "L1_prob": hold_proba_l1,
    "GB_prob": hold_proba_gb
})
preds.to_csv("holdout_predictions.csv", index=False)
print("Kaydedildi: holdout_predictions.csv")
display(preds.head())

import joblib

# LassoCV (28 özellikli) model ve kullandığı sütunları birlikte sakla
joblib.dump(
    {"model": lasso_model, "features": final_lasso_features},
    "lasso_cv_model.joblib"
)

# (İstersen) holdout tahminlerini de kaydet
lasso_hold_pred = pd.DataFrame({
    "Lasso_prob": y_hold_pred_en_cont if 'y_hold_pred_en_cont' in locals() else lasso_model.predict(X_holdout_fs.values)
})
lasso_hold_pred.to_csv("holdout_lasso_predictions.csv", index=False)
print("Kaydedildi: lasso_cv_model.joblib ve holdout_lasso_predictions.csv")
