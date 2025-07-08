import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv('train.csv')

# Load test data (no SalePrice column)
test_df = pd.read_csv('test.csv')

print(train_df.head())
print(test_df.head())

print("Train shape:", train_df.shape)
print(train_df.head())  



print(train_df.columns)  # Includes SalePrice (target)
print(test_df.columns)

# 1. Data Exploration
# 1. Data Exploration
print(train_df.info())
print(train_df.describe())


# 2. Handle missing values (simple example)
imputer = SimpleImputer(strategy='median')
df_numeric = train_df.select_dtypes(include=[np.number])
df_numeric_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns)

# 3. Handle categorical variables
df_cat = train_df.select_dtypes(include=['object'])
df_cat.fillna('Missing', inplace=True)
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
df_cat_encoded = pd.DataFrame(encoder.fit_transform(df_cat), columns=encoder.get_feature_names_out(df_cat.columns))

# 4. Combine processed data
df_processed = pd.concat([df_numeric_imputed, df_cat_encoded], axis=1)

# Target variable
y = df_processed['SalePrice']
X = df_processed.drop('SalePrice', axis=1)

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Model training
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Predictions and evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'RMSE: {rmse}')

# 8. Visualization
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices')
plt.show()
