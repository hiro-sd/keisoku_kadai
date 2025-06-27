import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# データの読み込み
df = pd.read_csv("/Users/yoshidahiroto/Downloads/修士関連/task.csv", parse_dates=["日付"])

# データ準備（説明変数と目的変数）
X = df[["ジム", "最高気温", "降水量", "歩数"]].values
y = df["作業時間"].values

# 標準化
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

results = {}

# 1. 線形回帰
lr = LinearRegression()
lr.fit(X_std, y)
y_pred_lr = lr.predict(X_std)
results["Linear"] = {
    "R2": r2_score(y, y_pred_lr),
    "RMSE": np.sqrt(mean_squared_error(y, y_pred_lr))
}

# 2. 多項式回帰（2次）
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_std)
lr_poly = LinearRegression()
lr_poly.fit(X_poly, y)
y_pred_poly = lr_poly.predict(X_poly)
results["Polynomial (deg2)"] = {
    "R2": r2_score(y, y_pred_poly),
    "RMSE": np.sqrt(mean_squared_error(y, y_pred_poly))
}

# 3. Ridge回帰
ridge = Ridge(alpha=1.0)
ridge.fit(X_std, y)
y_pred_ridge = ridge.predict(X_std)
results["Ridge"] = {
    "R2": r2_score(y, y_pred_ridge),
    "RMSE": np.sqrt(mean_squared_error(y, y_pred_ridge))
}

# 4. Lasso回帰
lasso = Lasso(alpha=0.1, max_iter=10000)
lasso.fit(X_std, y)
y_pred_lasso = lasso.predict(X_std)
results["Lasso"] = {
    "R2": r2_score(y, y_pred_lasso),
    "RMSE": np.sqrt(mean_squared_error(y, y_pred_lasso))
}

# 5. ランダムフォレスト回帰
rf = RandomForestRegressor(random_state=0, n_estimators=100)
rf.fit(X_std, y)
y_pred_rf = rf.predict(X_std)
results["Random Forest"] = {
    "R2": r2_score(y, y_pred_rf),
    "RMSE": np.sqrt(mean_squared_error(y, y_pred_rf))
}

# # 6. サポートベクター回帰
# svr = SVR()
# svr.fit(X_std, y)
# y_pred_svr = svr.predict(X_std)
# results["SVR"] = {
#     "R2": r2_score(y, y_pred_svr),
#     "RMSE": np.sqrt(mean_squared_error(y, y_pred_svr))
# }

# # 7. ElasticNet回帰
# elastic = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
# elastic.fit(X_std, y)
# y_pred_elastic = elastic.predict(X_std)
# results["ElasticNet"] = {
#     "R2": r2_score(y, y_pred_elastic),
#     "RMSE": np.sqrt(mean_squared_error(y, y_pred_elastic))
# }

# # 8. 主成分回帰（PCR: PCA+線形回帰, 2主成分で例示）
# pca_pcr = PCA(n_components=2)
# X_pcr = pca_pcr.fit_transform(X_std)
# lr_pcr = LinearRegression()
# lr_pcr.fit(X_pcr, y)
# y_pred_pcr = lr_pcr.predict(X_pcr)
# results["PCR (2PCs)"] = {
#     "R2": r2_score(y, y_pred_pcr),
#     "RMSE": np.sqrt(mean_squared_error(y, y_pred_pcr))
# }

# 結果表示
print("モデル比較（R2が高いほど説明力が高い、RMSEは小さいほど良い）")
for name, res in results.items():
    print(f"{name:20s}  R2: {res['R2']:.3f}  RMSE: {res['RMSE']:.2f}")