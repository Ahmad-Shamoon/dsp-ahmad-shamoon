import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_log_error
import joblib
import os
from house_prices.preprocess import clean_and_prepare_data


def build_model(data: pd.DataFrame) -> dict[str, float]:
    """
    Trains and saves Ridge model, encoder, and scaler.
    Returns performance metrics as a dictionary.
    """

    continuous_feats = ["GrLivArea", "GarageArea"]
    categorical_feats = ["MSZoning", "HouseStyle"]
    target_col = "SalePrice"

    # Clean data
    data = clean_and_prepare_data(data, continuous_feats, categorical_feats)

    # Split dataset
    X = data[continuous_feats + categorical_feats]
    y = data[target_col]
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing
    encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    scaler = StandardScaler()

    X_train_cat = encoder.fit_transform(X_train[categorical_feats])
    X_valid_cat = encoder.transform(X_valid[categorical_feats])
    X_train_num = scaler.fit_transform(X_train[continuous_feats])
    X_valid_num = scaler.transform(X_valid[continuous_feats])

    X_train_proc = np.concatenate([X_train_num, X_train_cat], axis=1)
    X_valid_proc = np.concatenate([X_valid_num, X_valid_cat], axis=1)

    # Model training
    model = Ridge(alpha=10.0, random_state=42)
    model.fit(X_train_proc, y_train)

    # Evaluation
    y_pred = np.maximum(model.predict(X_valid_proc), 1.0)
    rmsle = np.sqrt(mean_squared_log_error(y_valid, y_pred))

    # Save objects
    os.makedirs("../models", exist_ok=True)
    joblib.dump(model, "../models/model.joblib")
    joblib.dump(encoder, "../models/encoder.joblib")
    joblib.dump(scaler, "../models/scaler.joblib")

    print(f"✅ Model trained and saved. Validation RMSLE: {rmsle:.5f}")
    return {"rmsle": round(rmsle, 5)}
