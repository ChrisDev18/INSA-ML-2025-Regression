import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Load the datasets
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("/test.csv")

# Basic information about the datasets
train_info = train_df.info()
test_info = test_df.info()

# First few rows
train_head = train_df.head()
test_head = test_df.head()

# Shape of datasets
train_shape = train_df.shape
test_shape = test_df.shape

# Missing values
train_missing = train_df.isnull().sum()
test_missing = test_df.isnull().sum()

# Target variable distribution
target_description = train_df['co2'].describe()

"""# Data cleaning and preprocessing"""

# Drop columns with too many missing values and problematic categorical columns
cols_to_drop = ['hc', 'model']  # 'model' causes unseen labels problem
train_clean = train_df.drop(columns=cols_to_drop)
test_clean = test_df.drop(columns=cols_to_drop)

# Fill missing values for important columns
num_cols = ['co', 'nox', 'hcnox', 'ptcl', 'urb_cons', 'exturb_cons']

for col in num_cols:
    median_value = train_clean[col].median()
    train_clean[col].fillna(median_value, inplace=True)
    test_clean[col].fillna(median_value, inplace=True)

# Encode categorical variables
cat_cols = ['brand', 'car_class', 'range', 'fuel_type', 'hybrid', 'grbx_type_ratios']

encoders = {}
for col in cat_cols:
    encoder = LabelEncoder()
    train_clean[col] = encoder.fit_transform(train_clean[col])
    test_clean[col] = encoder.transform(test_clean[col])  # Same labels in test set
    encoders[col] = encoder

test_clean

train_clean = pd.get_dummies(train_clean, columns=cat_cols)
train_clean

test_clean = pd.get_dummies(test_clean, columns=cat_cols)
test_clean = test_clean.reindex(columns=train_clean.columns, fill_value=0)

test_clean

[x for x in train_clean.columns]

# Define Features and Target
X = train_clean.drop(columns=['id', 'co2'])
y = train_clean['co2']

X_test = test_clean.drop(columns=['id', 'co2'])
y_test = test_clean['co2']

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Check final shape
X.shape, X_test.shape, y.shape

"""# Model training and evaluation

## Linear regression
"""

# Split data into Training and Validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict on Validation set
y_pred = lr_model.predict(X_val)

# Evaluate using Mean Absolute Error (MAE)
mae = mean_absolute_error(y_val, y_pred)

mae

"""
On average, the model is off by 1.34 g/km of CO2 emission on the validation data. This is very good! Because if you look back at the target distribution:
	•	The mean CO2 emission is around 201 g/km.
	•	So, being off by only 1.34 is a small error."""

# Plot Real vs Predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_pred, alpha=0.3)
plt.xlabel('Real CO2 Emissions')
plt.ylabel('Predicted CO2 Emissions')
plt.title('Real vs Predicted CO2 Emissions (Validation Set)')
plt.grid(True)
plt.show()

"""#### Application of the model to the Test dataset"""

# Predict CO2 emissions on the Test dataset
test_predictions = lr_model.predict(X_test_scaled)

# Load the sample submission file
sample_submission = pd.read_csv("./Datasets/sample_submission.csv")

# Replace the 'co2' column with our predictions
sample_submission['co2'] = test_predictions

# Save the submission file
submission_path = "./SubmissionFiles/submission_linear_regression.csv"
sample_submission.to_csv(submission_path, index=False)

"""## Random Forest Regressor"""

# Create and train the Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=100,  # Number of trees
    max_depth=None,    # Trees can grow until all leaves are pure
    random_state=9,   # For reproducibility
    n_jobs=-1          # Use all CPU cores for faster computation
)
rf_model.fit(X_train, y_train)

# Predict on Validation set
y_pred_rf = rf_model.predict(X_val)

# Evaluate with Mean Absolute Error
mae_rf = mean_absolute_error(y_val, y_pred_rf)

mae_rf

# Create the submission file
rf_submission_path = "./SubmissionFiles/submission_random_forest.csv"
sample_submission.to_csv(rf_submission_path, index=False)

rf_submission_path

