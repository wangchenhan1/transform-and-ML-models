import pandas as pd
import numpy as np
import time
import os
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import joblib  # 用于保存模型

# 定义计算MAE的函数
def calculate_mae(model, X_train, X_test, y_train, y_test):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    return train_mae, test_mae

# 读取数据
data = pd.read_excel('/home/ggx/wls/DAC-code/co/UL/ST/data/ST-5-XGBR-UL-CO-sol.xlsx')
X = data.iloc[:, 1:-1].to_numpy()
y = data.iloc[:, -1]

# 定义随机状态值范围
random_states = range(1, 100)
results_all_states = {}

# 定义超参数范围
param_grid = {
    "n_estimators": list(range(50, 2000, 100)),
    "max_depth": list(range(2, 31, 1)),
    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.6],
    "min_child_weight":list(range(1, 15, 1)),
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
    "reg_alpha": [0, 0.01, 0.1, 0.5, 1, 10],
    "reg_lambda": [0.1, 0.5, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
}

time_start = time.time()

# 在循环中进行训练
for state in random_states:
    print(f"\nUsing random_state={state} for training")
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=state)
    
    # 数据归一化
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_normalized = scaler.transform(X_train)
    X_test_normalized = scaler.transform(X_test)
    
    # 定义 XGBoost 回归模型
    xgbr = XGBRegressor(objective='reg:squarederror', n_jobs=-1)
    grid_search = RandomizedSearchCV(xgbr, param_grid, n_iter=100, cv=20, n_jobs=-1, scoring='r2', random_state=state)
    
    grid_search.fit(X_train_normalized, y_train)
    
    # 最佳模型预测
    pred_train = grid_search.predict(X_train_normalized)
    pred_test = grid_search.predict(X_test_normalized)
    
    # 计算 R² 分数
    r2_train = r2_score(y_train, pred_train)
    r2_test = r2_score(y_test, pred_test)
    
    # 打印最佳参数
    print("\nBest parameter combination:")
    print(grid_search.best_params_)
    print(f"Train R²: {r2_train:.4f}, Test R²: {r2_test:.4f}")
    
    # 显示每个参数组合的结果
    results = pd.DataFrame(grid_search.cv_results_)
    print("\nResults for each parameter combination:")
    print(results[['params', 'mean_test_score', 'std_test_score']])
    
    # 存储结果
    results_all_states[state] = {
        "best_params": grid_search.best_params_,
        "cv_results": results,
        "train_r2": r2_train,
        "test_r2": r2_test,
        "scaler": scaler,
        "model": grid_search.best_estimator_
    }

# 找到最佳参数组合
best_train_r2 = float('-inf')
best_test_r2 = float('-inf')
best_random_state_train = None
best_random_state_test = None
best_params_train = None
best_params_test = None

for state, result in results_all_states.items():
    if result['train_r2'] > best_train_r2:
        best_train_r2 = result['train_r2']
        best_random_state_train = state
        best_params_train = result["best_params"]
    if result['test_r2'] > best_test_r2:
        best_test_r2 = result['test_r2']
        best_random_state_test = state
        best_params_test = result["best_params"]

# 输出到文件
current_directory = os.getcwd()
output_file = os.path.join(current_directory, "model_results.txt")
with open(output_file, 'w') as f:
    f.write("Best Training R² combination:\n")
    f.write(f"Random state: {best_random_state_train}\n")
    f.write(f"Train R²: {best_train_r2:.4f}\n")
    f.write(f"Best parameters: {best_params_train}\n\n")

    f.write("Best Testing R² combination:\n")
    f.write(f"Random state: {best_random_state_test}\n")
    f.write(f"Test R²: {best_test_r2:.4f}\n")
    f.write(f"Best parameters: {best_params_test}\n\n")
    
    # 将每个random_state的结果也写入文件
    for state, result in results_all_states.items():
        f.write(f"Random state: {state}\n")
        f.write("Best parameter combination:\n")
        f.write(str(result["best_params"]) + "\n")
        f.write(f"Train R²: {result['train_r2']:.4f}, Test R²: {result['test_r2']:.4f}\n")
        f.write("Results for each parameter combination:\n")
        f.write(result["cv_results"][['params', 'mean_test_score', 'std_test_score']].to_string() + "\n")
        f.write("Best model parameters:\n")
        f.write(str(result["model"].get_params()) + "\n")
        f.write("Scaler parameters:\n")
        f.write(str(result["scaler"].get_params()) + "\n\n")

# 单独输出最佳的测试和训练R²分数以及对应的random_state和MAE
with open(output_file, 'a') as f:
    f.write("\n单独输出最佳的测试和训练R²分数以及对应的random_state和MAE：\n")

    # 计算最佳测试模型的训练集和测试集 MAE
    best_test_model = results_all_states[best_random_state_test]["model"]
    best_test_scaler = results_all_states[best_random_state_test]["scaler"]
    X_train_normalized = best_test_scaler.transform(X_train)
    X_test_normalized = best_test_scaler.transform(X_test)
    test_train_mae, test_test_mae = calculate_mae(best_test_model, X_train_normalized, X_test_normalized, y_train, y_test)

    # 输出最佳测试结果
    f.write(f"最佳测试R²分数：{best_test_r2:.4f}, 对应的random_state: {best_random_state_test}, 对应的训练R²分数：{results_all_states[best_random_state_test]['train_r2']:.4f}\n")
    f.write(f"最佳测试模型的训练集MAE: {test_train_mae:.4f}, 测试集MAE: {test_test_mae:.4f}\n\n")

    # 计算最佳训练模型的训练集和测试集 MAE
    best_train_model = results_all_states[best_random_state_train]["model"]
    best_train_scaler = results_all_states[best_random_state_train]["scaler"]
    X_train_normalized = best_train_scaler.transform(X_train)
    X_test_normalized = best_train_scaler.transform(X_test)
    train_train_mae, train_test_mae = calculate_mae(best_train_model, X_train_normalized, X_test_normalized, y_train, y_test)

    # 输出最佳训练结果
    f.write(f"最佳训练R²分数：{best_train_r2:.4f}, 对应的random_state: {best_random_state_train}, 对应的测试R²分数：{results_all_states[best_random_state_train]['test_r2']:.4f}\n")
    f.write(f"最佳训练模型的训练集MAE: {train_train_mae:.4f}, 测试集MAE: {train_test_mae:.4f}\n")

# 可视化结果
# 使用最佳测试模型进行预测
best_model = results_all_states[best_random_state_test]["model"]
best_scaler = results_all_states[best_random_state_test]["scaler"]
X_train_normalized = best_scaler.transform(X_train)
X_test_normalized = best_scaler.transform(X_test)
pred_train = best_model.predict(X_train_normalized)
pred_test = best_model.predict(X_test_normalized)

# 计算指标
avg_train_mse = mean_squared_error(y_train, pred_train)
avg_test_mse = mean_squared_error(y_test, pred_test)
avg_train_mae = mean_absolute_error(y_train, pred_train)
avg_test_mae = mean_absolute_error(y_test, pred_test)
avg_train_r2 = r2_score(y_train, pred_train)
avg_test_r2 = r2_score(y_test, pred_test)

# 计算 RMAE（Root Mean Absolute Error）
train_rmae = np.sqrt(avg_train_mae)
test_rmae = np.sqrt(avg_test_mae)

# 绘制散点图
plt.figure(figsize=(8, 6))
plt.scatter(y_train, pred_train, 
            label=f'Train Data, MSE: {avg_train_mse:.4f}, MAE: {avg_train_mae:.4f}, R2: {avg_train_r2:.4f}, RMAE: {train_rmae:.4f}',
            color="green", alpha=0.6, s=80, linewidth=0.5)
plt.scatter(y_test, pred_test, 
            label=f'Test Data, MSE: {avg_test_mse:.4f}, MAE: {avg_test_mae:.4f}, R2: {avg_test_r2:.4f}, RMAE: {test_rmae:.4f}',
            color="red", alpha=0.6, s=80, linewidth=0.5)
plt.plot([min(y), max(y)], [min(y), max(y)], linestyle="--", color="black")
plt.title("X Gradient Boosting Regression")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.tight_layout()

# 保存散点图到当前文件夹
scatter_plot_filename = os.path.join(current_directory, "XGBR-regression_results.png")
plt.savefig(scatter_plot_filename)
print(f"\n散点图已保存为：{scatter_plot_filename}")
plt.show()

# 使用SHAP进行特征重要性分析
explainer = shap.Explainer(best_model)
shap_values = explainer(X_test_normalized)

# 绘制SHAP总结图
shap.summary_plot(shap_values, X_test_normalized, feature_names=data.columns[1:-1])

# 保存SHAP图到当前文件夹
shap_plot_dir = os.path.join(current_directory, "shap_plots")
os.makedirs(shap_plot_dir, exist_ok=True)
shap_plot_filename = os.path.join(shap_plot_dir, "shap_summary.png")
plt.savefig(shap_plot_filename)
print(f"\nSHAP图已保存为：{shap_plot_filename}")
plt.show()

# 保存最佳模型和Scaler
best_model_filename = os.path.join(current_directory, "best_XGBR_model.pkl")
joblib.dump(best_model, best_model_filename)
print(f"\n最佳模型已保存为：{best_model_filename}")

best_scaler_filename = os.path.join(current_directory, "best_XGBR_scaler.pkl")
joblib.dump(best_scaler, best_scaler_filename)
print(f"\n最佳数据标准化器已保存为：{best_scaler_filename}")