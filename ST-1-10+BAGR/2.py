import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.ensemble import (
    RandomForestRegressor, 
    AdaBoostRegressor, 
    GradientBoostingRegressor, 
    BaggingRegressor  # 导入 BaggingRegressor
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import joblib

# 读取数据
data = pd.read_excel('/home/ggx/wls/DAC-code/co/UL/data/UL-CO-sol.xlsx')
X = data.iloc[:, 1:-1]  # 特征数据（假设第1列到倒数第2列是特征）
y = data.iloc[:, -1]    # 目标数据（假设最后一列是目标变量）

# Parameters for the Symbolic Transformer (ST)
population_size = 30000  # 增加种群大小
generations = 100        # 增加代数
function_set = ['add', 'sub', 'mul', 'div', 'abs', 'sqrt', 'log', 'inv', 'sin', 'cos']  # 增加函数集
n_components = np.arange(1, 11, 1)  # 1D to 10D
p_crossover = 0.7  # 增加交叉概率
p_subtree_mutation = 0.1  # 调整子树变异概率
parsimony_coefficient = 0.001  # 减小简约系数
p_hoist_mutation = 0.05  # 调整提升变异概率
p_point_mutation = 0.1  # 调整点变异概率
tournament_size = 20

# Define models
models = {
    'RandomForestRegressor': RandomForestRegressor(random_state=42),
    'AdaBoostRegressor': AdaBoostRegressor(random_state=42),
    'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42),
    'BaggingRegressor': BaggingRegressor(random_state=42),  # 添加 BaggingRegressor
    'XGBRegressor': XGBRegressor(use_label_encoder=False, eval_metric='rmse', random_state=42),
    'LGBMRegressor': LGBMRegressor(random_state=42),
    'CatBoostRegressor': CatBoostRegressor(random_state=42, verbose=0, allow_writing_files=False)
}

# Initialize dictionaries to store results
rmse_scores = {model: {n_dim: [] for n_dim in n_components} for model in models}
r2_scores = {model: {n_dim: [] for n_dim in n_components} for model in models}
correlations = {n_dim: [] for n_dim in n_components}
train_mae_scores = {model: {n_dim: [] for n_dim in n_components} for model in models}
test_mae_scores = {model: {n_dim: [] for n_dim in n_components} for model in models}

# Create an ExcelWriter object to save all sheets in one Excel file
excel_writer = pd.ExcelWriter('transformed_features_all_dimensions.xlsx', engine='xlsxwriter')

# Loop through different feature dimensions (1D to 10D)
for n_dim in n_components:
    # Initialize the Symbolic Transformer
    transformer = SymbolicTransformer(
        population_size=population_size,
        generations=generations,
        function_set=function_set,
        parsimony_coefficient=parsimony_coefficient,
        p_crossover=p_crossover,
        p_subtree_mutation=p_subtree_mutation,
        p_hoist_mutation=p_hoist_mutation,
        p_point_mutation=p_point_mutation,
        n_components=n_dim,
        tournament_size=tournament_size,
        const_range=(-1, 1),
        random_state=42
    )

    # Fit the symbolic transformer on the data
    transformer.fit(X, y)
    # Save the transformer to current folder with dynamic naming for each dimension
    joblib.dump(transformer, f'symbolic_transformer_{n_dim}D.pkl')  # Save each transformer with a unique name
    # Transform the features using the symbolic expressions
    X_transformed = transformer.transform(X)

    # Save the transformed features to an Excel file in a separate sheet
    transformed_data = pd.DataFrame(X_transformed, columns=[f'Feature_{i+1}' for i in range(n_dim)])
    transformed_data.to_excel(excel_writer, sheet_name=f'{n_dim}D', index=False)

    # Calculate the correlation between each transformed feature and the target variable
    for i in range(X_transformed.shape[1]):
        corr, _ = pearsonr(X_transformed[:, i], y)
        correlations[n_dim].append(corr)
        print(f"Dimension: {n_dim}D, Feature: {i+1}, Correlation with target: {corr}")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

    # Train and evaluate each model
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        r2 = r2_score(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)

        rmse_scores[model_name][n_dim].append(rmse)
        r2_scores[model_name][n_dim].append(r2)
        train_mae_scores[model_name][n_dim].append(train_mae)
        test_mae_scores[model_name][n_dim].append(test_mae)

        print(f"Model: {model_name}, Dimension: {n_dim}D, RMSE: {rmse}, R2: {r2}, Train MAE: {train_mae}, Test MAE: {test_mae}")

# Save the Excel file
excel_writer.close()

# Save metrics to a CSV file
metrics_data = []
for model_name in models:
    for n_dim in n_components:
        avg_r2 = np.mean(r2_scores[model_name][n_dim])
        avg_train_mae = np.mean(train_mae_scores[model_name][n_dim])
        avg_test_mae = np.mean(test_mae_scores[model_name][n_dim])
        metrics_data.append([model_name, n_dim, avg_r2, avg_train_mae, avg_test_mae])

metrics_df = pd.DataFrame(metrics_data, columns=['Model', 'Dimension', 'R2', 'Train MAE', 'Test MAE'])
metrics_df.to_csv('model_metrics.csv', index=False)

print("Metrics saved to 'model_metrics.csv'")

# Save R2 and MAE data to Excel
r2_df = pd.DataFrame(columns=['Model', 'Dimension', 'R2'])
train_mae_df = pd.DataFrame(columns=['Model', 'Dimension', 'Train MAE'])
test_mae_df = pd.DataFrame(columns=['Model', 'Dimension', 'Test MAE'])

for model_name in models:
    for n_dim in n_components:
        avg_r2 = np.mean(r2_scores[model_name][n_dim])
        avg_train_mae = np.mean(train_mae_scores[model_name][n_dim])
        avg_test_mae = np.mean(test_mae_scores[model_name][n_dim])
        
        r2_df = r2_df.append({'Model': model_name, 'Dimension': n_dim, 'R2': avg_r2}, ignore_index=True)
        train_mae_df = train_mae_df.append({'Model': model_name, 'Dimension': n_dim, 'Train MAE': avg_train_mae}, ignore_index=True)
        test_mae_df = test_mae_df.append({'Model': model_name, 'Dimension': n_dim, 'Test MAE': avg_test_mae}, ignore_index=True)

r2_df.to_excel('r2_vs_feature_dimension_by_model.xlsx', index=False)
train_mae_df.to_excel('train_mae_vs_feature_dimension_by_model.xlsx', index=False)
test_mae_df.to_excel('test_mae_vs_feature_dimension_by_model.xlsx', index=False)

print("R2 data saved to 'r2_vs_feature_dimension_by_model.xlsx'")
print("Train MAE data saved to 'train_mae_vs_feature_dimension_by_model.xlsx'")
print("Test MAE data saved to 'test_mae_vs_feature_dimension_by_model.xlsx'")

# Plot R2 vs. Feature Dimension for each model
plt.figure(figsize=(12, 8))
for model_name in models:
    r2_list = [np.mean(r2_scores[model_name][n_dim]) for n_dim in n_components]
    plt.plot(n_components, r2_list, marker='o', label=f'R2 - {model_name}')
plt.xlabel('Dimension of Features')
plt.ylabel('R2 Score')
plt.title('R2 vs. Feature Dimension (1D to 10D) for each model')
plt.grid(True)
plt.legend()
plt.savefig('r2_vs_feature_dimension_by_model.png')  # Save the plot to a file
plt.show()

# Plot Train MAE vs. Feature Dimension for each model
plt.figure(figsize=(12, 8))
for model_name in models:
    train_mae_list = [np.mean(train_mae_scores[model_name][n_dim]) for n_dim in n_components]
    plt.plot(n_components, train_mae_list, marker='o', label=f'Train MAE - {model_name}')
plt.xlabel('Dimension of Features')
plt.ylabel('Train MAE')
plt.title('Train MAE vs. Feature Dimension (1D to 10D) for each model')
plt.grid(True)
plt.legend()
plt.savefig('train_mae_vs_feature_dimension_by_model.png')  # Save the plot to a file
plt.show()

# Plot Test MAE vs. Feature Dimension for each model
plt.figure(figsize=(12, 8))
for model_name in models:
    test_mae_list = [np.mean(test_mae_scores[model_name][n_dim]) for n_dim in n_components]
    plt.plot(n_components, test_mae_list, marker='o', label=f'Test MAE - {model_name}')
plt.xlabel('Dimension of Features')
plt.ylabel('Test MAE')
plt.title('Test MAE vs. Feature Dimension (1D to 10D) for each model')
plt.grid(True)
plt.legend()
plt.savefig('test_mae_vs_feature_dimension_by_model.png')  # Save the plot to a file
plt.show()

# Save R2 and MAE data to Excel
r2_df = pd.DataFrame(columns=['Model', 'Dimension', 'R2'])
train_mae_df = pd.DataFrame(columns=['Model', 'Dimension', 'Train MAE'])
test_mae_df = pd.DataFrame(columns=['Model', 'Dimension', 'Test MAE'])

for model_name in models:
    for n_dim in n_components:
        avg_r2 = np.mean(r2_scores[model_name][n_dim])
        avg_train_mae = np.mean(train_mae_scores[model_name][n_dim])
        avg_test_mae = np.mean(test_mae_scores[model_name][n_dim])
        
        r2_df = r2_df.append({'Model': model_name, 'Dimension': n_dim, 'R2': avg_r2}, ignore_index=True)
        train_mae_df = train_mae_df.append({'Model': model_name, 'Dimension': n_dim, 'Train MAE': avg_train_mae}, ignore_index=True)
        test_mae_df = test_mae_df.append({'Model': model_name, 'Dimension': n_dim, 'Test MAE': avg_test_mae}, ignore_index=True)

# Save the dataframes to Excel files
r2_df.to_excel('r2_vs_feature_dimension_by_model.xlsx', index=False)
train_mae_df.to_excel('train_mae_vs_feature_dimension_by_model.xlsx', index=False)
test_mae_df.to_excel('test_mae_vs_feature_dimension_by_model.xlsx', index=False)

print("R2 data saved to 'r2_vs_feature_dimension_by_model.xlsx'")
print("Train MAE data saved to 'train_mae_vs_feature_dimension_by_model.xlsx'")
print("Test MAE data saved to 'test_mae_vs_feature_dimension_by_model.xlsx'")