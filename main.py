from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from read import read_data, process_data
from test import test_model

test_data = read_data("data/test.csv")
train_data = read_data("data/train.csv")

train_unsplit, train_split, val_split, x_test = process_data(train_data, test_data)
x_train_val, y_train_val = train_unsplit    # All given examples
x_train, y_train = train_split              # The training examples
x_val, y_val = val_split                    # The validation examples

# # Create and train Linear Regression model
# reg_model = LinearRegression().fit(x_train, y_train)
#
# # Predict on Validation set
# y_pred = reg_model.predict(x_val)
#
# # Evaluate using Mean Absolute Error (MAE)
# mae = mean_absolute_error(y_val, y_pred)
# print("Mean absolute error:", mae)
#
# # Update estimator using all examples to improve performance
# reg_model.fit(x_train_val, y_train_val)
#
# # Perform inferencing on test data and save to CSV
# test_model(reg_model, x_test, test_data["id"], save_path="submission_reg.csv")

# Create and train the Random Forest model
rand_forest_model = RandomForestRegressor(
    n_estimators=100,   # Number of trees
    max_depth=None,     # Trees can grow until all leaves are pure
    random_state=31,     # For reproducibility
    n_jobs=-1           # Use all CPU cores for faster computation
).fit(x_train, y_train)

# Predict on Validation set
y_pred = rand_forest_model.predict(x_val)

# Evaluate using Mean Absolute Error (MAE)
mae = mean_absolute_error(y_val, y_pred)
print("Mean absolute error:", mae)

# Update estimator using all examples to improve performance
rand_forest_model.fit(x_train_val, y_train_val)

# Perform inferencing on test data and save to CSV
test_model(rand_forest_model, x_test, test_data["id"], save_path="submission_rand_tree.csv")