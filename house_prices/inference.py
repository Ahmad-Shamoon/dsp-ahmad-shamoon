import pandas as pd
import numpy as np
import joblib
from house_prices.preprocess import clean_and_prepare_data

def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    """
    Loads saved model and preprocessing objects to predict SalePrice.
    """
    categorical_feats = ["MSZoning", "HouseStyle"]
    continuous_feats = ["GrLivArea", "GarageArea"]

    model = joblib.load("../models/model.joblib")
    encoder = joblib.load("../models/encoder.joblib")
    scaler = joblib.load("../models/scaler.joblib")

    input_data = clean_and_prepare_data(input_data, continuous_feats, categorical_feats)

    X_cat = encoder.transform(input_data[categorical_feats])
    X_num = scaler.transform(input_data[continuous_feats])
    X_proc = np.concatenate([X_num, X_cat], axis=1)

    preds = np.maximum(model.predict(X_proc), 1.0)
    return preds
