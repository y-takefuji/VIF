import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score

# ------------------------------------------------------------
# 1. データ読み込み
# ------------------------------------------------------------
file_path = "Supplementary Data 1 – Database.xlsx"
df = pd.read_excel(file_path)

# 列名の前後の空白を除去
df.columns = [str(c).strip() for c in df.columns]

# ------------------------------------------------------------
# 2. データセットの形状とターゲット分布
# ------------------------------------------------------------
print("Shape of dataset (raw):", df.shape)

target_col = "PI2"
if target_col not in df.columns:
    # 大文字小文字の違いに対応
    cand = [c for c in df.columns if c.lower() == "target"]
    if not cand:
        raise ValueError("'target' 列が見つかりません。列名を確認してください: " + str(list(df.columns)))
    target_col = cand[0]

print("\nTarget distribution (counts):")
print(df[target_col].value_counts(dropna=False).sort_index())
print("\nTarget distribution (ratio):")
print(df[target_col].value_counts(normalize=True, dropna=False).sort_index().round(4))

# ------------------------------------------------------------
# 3. 変数の削除（最初の変数, PI3, PI4）
# ------------------------------------------------------------
# 「最初の変数」= ターゲット以外で先頭にある列（ID等を想定）
first_var = [c for c in df.columns if c != target_col][0]
drop_cols = [first_var, "PI3", "PI4"]
drop_cols = [c for c in drop_cols if c in df.columns]
print("\nDropped columns:", drop_cols)

df = df.drop(columns=drop_cols)

# すべて数値として扱う
df = df.apply(pd.to_numeric, errors="coerce")
df = df.dropna().reset_index(drop=True)

print("\nShape of dataset (after drop):", df.shape)

y = df[target_col]
X = df.drop(columns=[target_col])
print("Features:", list(X.columns))

# ------------------------------------------------------------
# 4. VIF（分散拡大係数）
# ------------------------------------------------------------
X_const = add_constant(X)  # 切片を追加
vif_values = []
for i, col in enumerate(X.columns):
    vif_values.append(variance_inflation_factor(X_const.values, i + 1))  # 0列目は定数

vif_df = pd.DataFrame({"variable": X.columns, "VIF": vif_values})

# ------------------------------------------------------------
# 5. RFRI（ランダムフォレスト冗長性指標）
#    各変数を他の全変数からRFで予測し、
#    OOF(交差検証)予測のR2を冗長性として算出
#    RFRI = R2（1に近いほど他変数で再現可能＝冗長）
#    併せてVIF相当の 1/(1-R2) も参考として算出
# ------------------------------------------------------------
rfri_values = []
for col in X.columns:
    X_other = X.drop(columns=[col])
    y_col = X[col]

    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    pred = cross_val_predict(rf, X_other, y_col, cv=5)
    r2 = r2_score(y_col, pred)
    r2 = max(r2, 0.0)  # 負のR2は0扱い
    rfri_values.append(r2)

rfri_df = pd.DataFrame({"variable": X.columns, "RFRI": rfri_values})

# ------------------------------------------------------------
# 6. 集計テーブルの作成と保存
# ------------------------------------------------------------
result = vif_df.merge(rfri_df, on="variable")
result = result.sort_values("VIF", ascending=False).reset_index(drop=True)

print("\nSummary table:")
print(result.round(4))

result.to_csv("result.csv", index=False, encoding="utf-8-sig")
print("\nSaved: result.csv")