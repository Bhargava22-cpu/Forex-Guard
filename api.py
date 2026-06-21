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

model = joblib.load("if_model.pkl")
scaler = joblib.load("scaler.pkl")
lstm_model = load_model("lstm_model.keras")
lstm_threshold = joblib.load("lstm_threshold.pkl")

r = fakeredis.FakeRedis()
HISTORY_TTL = 86400


def _get(key: str) -> list:
    raw = r.get(key)
    return json.loads(raw) if raw else []


def _set(key: str, value: list):
    r.set(key, json.dumps(value, default=str))
    r.expire(key, HISTORY_TTL)


def get_top_if_features(X: pd.DataFrame, top_n: int = 3) -> list[dict]:
    baseline = model.decision_function(X)[0]
    impacts = {}

    for col in feature_cols:
        X_perturbed = X.copy()
        X_perturbed[col] = 0.0
        perturbed_score = model.decision_function(X_perturbed)[0]
        impacts[col] = round(float(baseline - perturbed_score), 6)

    top = sorted(impacts.items(), key=lambda x: -abs(x[1]))[:top_n]
    return [{"feature": k, "impact": v} for k, v in top]


class EventInput(BaseModel):
    user_id: int
    timestamp: str
    event_type: str
    trade_volume: float = 0
    amount: float = 0
    margin: float = 0
    instrument: str = "NONE"
    ip_address: str


@app.get("/")
def home():
    return {"message": "API is running"}


@app.post(
    "/predict",
    summary="Predict anomaly",
    description="Returns anomaly score, label, top features, and explanation"
)
def predict(data: EventInput):

    event_timestamp = datetime.fromisoformat(data.timestamp)

    history = _get(str(data.user_id))

    for event in history:
        if isinstance(event["timestamp"], str):
            event["timestamp"] = datetime.fromisoformat(event["timestamp"])

    event_dict = data.dict()
    event_dict["timestamp"] = event_timestamp

    features = compute_features(event_dict, history)

    df = pd.DataFrame([features])

    feature_key = f"{data.user_id}_features"
    feature_history = _get(feature_key)
    feature_history.append(df.iloc[0].to_dict())
    feature_history = feature_history[-10:]
    _set(feature_key, feature_history)

    X = df[feature_cols].fillna(0)

    score = model.decision_function(X)[0]
    pred = model.predict(X)[0]
    pred = 0 if pred == 1 else 1

    risk_score = max(0.0, min(1.0, (1 - score) / 2))

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

    final_anomaly = int(pred == 1 or lstm_anomaly == 1)

    top_features = get_top_if_features(X) if final_anomaly == 1 else []
    reason = generate_reason(top_features)

    history.append({
        "timestamp": str(event_timestamp),
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