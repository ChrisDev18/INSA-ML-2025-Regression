import pandas as pd
from sklearn.base import BaseEstimator


def test_model(model, test_data, ids, save_path="submission.csv"):
    # Make predictions on a test set
    y_preds = model.predict(test_data)

    # Save predictions to CSV
    submission = pd.DataFrame({"id": ids, "co2": y_preds})
    submission.to_csv(save_path, index=False)