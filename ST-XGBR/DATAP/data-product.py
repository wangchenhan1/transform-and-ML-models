# run_all_final.py
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
from sklearn.pipeline import Pipeline

# ===== 全局字体 & 中文字体设置 =====
rcParams['font.family'] = 'Arial'
rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 12})

# ------------------------------------------------------------
# 1. 读取数据
# ------------------------------------------------------------
data = pd.read_excel('/home/ggx/wls/DAC-code/co/UL/w+CrCr/ST/data/ST-5-XGBR-UL-CO-sol.xlsx')
X = data.iloc[:, 1:-1].to_numpy()
y = data.iloc[:, -1].to_numpy()
feature_names = data.columns[1:-1].tolist()

# ------------------------------------------------------------
# 2. 载入模型 & Scaler
# ------------------------------------------------------------
best_model = joblib.load('/home/ggx/wls/DAC-code/co/UL/w+CrCr/ST/ST-XGBR/best_XGBR_model.pkl')
best_scaler = joblib.load('/home/ggx/wls/DAC-code/co/UL/w+CrCr/ST/ST-XGBR/best_XGBR_scaler.pkl')

# 创建Pipeline用于正确的PDP计算
pipeline = Pipeline([
    ('scaler', best_scaler),
    ('model', best_model)
])

# ------------------------------------------------------------
# 3. 数据划分 & 归一化
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_norm = best_scaler.transform(X_train)
X_test_norm = best_scaler.transform(X_test)

# ------------------------------------------------------------
# 4. 预测 & 评估
# ------------------------------------------------------------
pred_train = best_model.predict(X_train_norm)
pred_test = best_model.predict(X_test_norm)

print(f"Train MSE: {mean_squared_error(y_train, pred_train):.4f}, "
      f"Train MAE: {mean_absolute_error(y_train, pred_train):.4f}, "
      f"Train R2:  {r2_score(y_train, pred_train):.4f}")
print(f"Test  MSE: {mean_squared_error(y_test, pred_test):.4f}, "
      f"Test  MAE: {mean_absolute_error(y_test, pred_test):.4f}, "
      f"Test  R2:  {r2_score(y_test, pred_test):.4f}")

# ------------------------------------------------------------
# 5. 创建散点图数据并保存到Excel
# ------------------------------------------------------------
# 创建包含实际值和预测值的DataFrame
scatter_data = pd.DataFrame({
    'Set': ['Train'] * len(y_train) + ['Test'] * len(y_test),
    'Actual': np.concatenate([y_train, y_test]),
    'Predicted': np.concatenate([pred_train, pred_test])
})

# 保存到Excel文件
scatter_data.to_excel('scatter_plot_data.xlsx', index=False)
print("散点图数据已保存到 scatter_plot_data.xlsx")

# ------------------------------------------------------------
# 6. 回归散点图
# ------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(y_train, pred_train, label='Train', color='green', alpha=0.6, s=80)
plt.scatter(y_test, pred_test, label='Test', color='red', alpha=0.6, s=80)
lims = [min(y.min(), pred_train.min(), pred_test.min()),
        max(y.max(), pred_train.max(), pred_test.max())]
plt.plot(lims, lims, 'k--', lw=1)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Regression Results')
plt.legend()
plt.tight_layout()
plt.savefig('regression_results.png', dpi=300)
plt.close()

print("回归散点图已保存为 regression_results.png")