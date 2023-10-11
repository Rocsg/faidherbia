import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Load the CSV file into a DataFrame
data = pd.read_csv('/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/DeepEstimator/Data/Full_dataset/TargetValues/Data_faidherbia.csv')
# Remove rows with NaN values in the target variables (columns 8, 9, and 10)
data.dropna(subset=['#TARGET-Yield', '#TARGET-biomass', '#TARGET-LAI'], inplace=True)
# Separate the features (columns 4, 5, 6, and 7) and the target variables (columns 8, 9, and 10)
X = data[['#PRM-Rh_air', '#PRM-Solar_azimuth', '#PRM-PAR', '#PRM-Tsoil']]
#y = data[['#TARGET-Yield', '#TARGET-biomass', '#TARGET-LAI']]
y = data[['#TARGET-LAI']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# Train the model on the training data
rf_model.fit(X_train, y_train)
# Make predictions on the test data
predictions = rf_model.predict(X_test)


xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
# Entraîner le modèle sur les données d'entraînement
xgb_model.fit(X_train, y_train)
# Faire des prédictions sur les données de test
predictions = xgb_model.predict(X_test)



# Evaluate the model's performance (you can use different metrics depending on your problem)
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Extract the correlation coefficients
import scipy.stats as stats
corr_col8, _ = stats.pearsonr(predictions, y_test)
print(corr_col8)
