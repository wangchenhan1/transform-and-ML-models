#迭代1000次取平均值
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
# Load the saved model
best_model = joblib.load('/home/ggx/wls/DAC-code/CR-X/HCOOH-C/BAGR/ST/2D/best_bagging_model.pkl')
best_scaler = joblib.load('/home/ggx/wls/DAC-code/CR-X/HCOOH-C/BAGR/ST/2D/best_bagging_scaler.pkl')
print("Model loaded successfully!")

# Load new data
new_data = pd.read_excel("/home/ggx/wls/DAC-code/CR-X/HCOOH-C/BAGR/ST/2D/predict/transformed_combined_features_2D.xlsx")  # Example of how to load new data
new_data 

# Select features (make sure the new data has the same feature columns as the training data)
X_new = new_data.iloc[:, 1:].to_numpy()  # Example assuming the first column is not part of features

# Normalize the new data
scaler = MinMaxScaler()
X_new_normalized = scaler.fit_transform(X_new)  # Ensure the same normalization as training data

# Initialize an array to store predictions for each iteration
predictions_all = np.zeros((X_new.shape[0], 1000))  # Shape: (number of samples, 1000)

# Perform 1000 iterations of predictions
for i in range(1000):
    predictions = best_model.predict(X_new_normalized)
    predictions_all[:, i] = predictions

# Calculate the mean of predictions across all iterations
final_predictions = np.mean(predictions_all, axis=1)

# Add final predictions as a new column to the DataFrame (e.g., "Predictions")
new_data['Predictions'] = final_predictions

# Output the updated DataFrame with predictions to an Excel file
output_file = "/home/ggx/wls/DAC-code/CR-X/HCOOH-C/BAGR/ST/2D/predict/predict_BAR-output-10ST-hcooh-C-2.xlsx"
new_data.to_excel(output_file, index=False)

print(f"Predictions saved to {output_file}")