from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import json

import fakeredis
from tensorflow.keras.models import load_model

from feature_engineering import compute_features, feature_cols
from reason import generate_reason

app = FastAPI(
    title="Forex Anomaly Detection API",
    description="Detects anomalies using Isolation Forest and LSTM Autoencoder",
    version="1.0"
)

# Load model + scaler + lstm
model = joblib.load("if_model.pkl")
scaler = joblib.load("scaler.pkl")
lstm_model = load_model("lstm_model.keras")
lstm_threshold = joblib.load("lstm_threshold.pkl")

# Fakeredis
r = fakeredis.FakeRedis()
HISTORY_TTL = 86400  # 24 hours


def _get(key: str) -> list:
    raw = r.get(key)
    return json.loads(raw) if raw else []


def _set(key: str, value: list):
    r.set(key, json.dumps(value, default=str))
    r.expire(key, HISTORY_TTL)


def get_top_if_features(X_scaled: pd.DataFrame, top_n: int = 3) -> list[dict]:
    baseline = model.decision_function(X_scaled)[0]
    impacts = {}

    for col in feature_cols:
        X_perturbed = X_scaled.copy()
        X_perturbed[col] = 0.0
        perturbed_score = model.decision_function(X_perturbed)[0]
        impacts[col] = round(float(baseline - perturbed_score), 6)

    top = sorted(impacts.items(), key=lambda x: -abs(x[1]))[:top_n]
    return [{"feature": k, "impact": v} for k, v in top]


# Pydantic input schema
class EventInput(BaseModel):
    user_id: int
    timestamp: str
    event_type: str
    trade_volume: float = 0
    amount: float = 0
    margin: float = 0
    instrument: str = "NONE"
    ip_address: str


# Health check endpoint
@app.get("/")
def home():
    return {"message": "API is running"}


# Prediction endpoint
@app.post(
    "/predict",
    summary="Predict anomaly",
    description="Returns anomaly score, label, top features, and explanation"
)
def predict(data: EventInput):

    # Convert timestamp string to datetime
    data.timestamp = datetime.fromisoformat(data.timestamp)

    # Get raw event history for this user from Redis
    history = _get(str(data.user_id))

    # Convert timestamp strings back to datetime for feature engineering
    for event in history:
        if isinstance(event["timestamp"], str):
            event["timestamp"] = datetime.fromisoformat(event["timestamp"])

    # Compute engineered features
    features = compute_features(data, history)

    # Convert features → DataFrame
    df = pd.DataFrame([features])

    # Store engineered feature history for LSTM
    feature_key = f"{data.user_id}_features"
    feature_history = _get(feature_key)
    feature_history.append(df.iloc[0].to_dict())
    feature_history = feature_history[-10:]
    _set(feature_key, feature_history)

    # Ensure feature order
    X = df[feature_cols].fillna(0)

    # Scale
    X_scaled = pd.DataFrame(
        scaler.transform(X),
        columns=feature_cols
    )

    # Isolation Forest prediction
    score = model.decision_function(X_scaled)[0]
    pred = model.predict(X_scaled)[0]
    pred = 0 if pred == 1 else 1

    # Normalize score to 0–1
    risk_score = max(0.0, min(1.0, (1 - score) / 2))

    # LSTM prediction
    if len(feature_history) == 10:
        seq_df = pd.DataFrame(feature_history)
        X_seq = seq_df[feature_cols].fillna(0)
        X_seq = scaler.transform(X_seq)
        X_seq = X_seq.reshape(1, len(feature_history), len(feature_cols))
        reconstruction = lstm_model.predict(X_seq, verbose=0)
        lstm_error = float(((X_seq - reconstruction) ** 2).mean())
        lstm_anomaly = int(lstm_error > lstm_threshold)
    else:
        lstm_anomaly = 0

    # Final decision
    final_anomaly = int(pred == 1 or lstm_anomaly == 1)

    # Reason derived from top features
    top_features = get_top_if_features(X_scaled) if final_anomaly == 1 else []
    reason = generate_reason(top_features)

    # Save current event into history
    history.append({
        "timestamp": str(data.timestamp),
        "event_type": data.event_type,
        "trade_volume": data.trade_volume,
        "amount": data.amount,
        "margin": data.margin,
        "instrument": data.instrument,
        "ip_address": data.ip_address
    })
    _set(str(data.user_id), history[-10:])

    return {
        "final_anomaly": int(final_anomaly),
        "risk_score": round(float(risk_score), 4),
        "reason": reason
    }
