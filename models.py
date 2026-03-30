import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense

from feature_engineering import feature_cols
from reason import generate_reason


# Load and sort
df = pd.read_csv("engineered_features.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)


# Time-based split
split_idx = int(0.8 * len(df))
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

X_train = train_df[feature_cols].fillna(0)
X_test = test_df[feature_cols].fillna(0)


# Isolation Forest
model = IsolationForest(
    n_estimators=100,
    contamination=0.03,
    random_state=42
)

model.fit(X_train)

# Save model
joblib.dump(model, "if_model.pkl")

test_df["if_score"] = model.decision_function(X_test)
test_df["if_anomaly"] = model.predict(X_test)
test_df["if_anomaly"] = test_df["if_anomaly"].map({1: 0, -1: 1})

print("Isolation Forest Results:")
print(classification_report(
    test_df["is_anomaly"],
    test_df["if_anomaly"]
))


# LSTM Autoencoder

# Fit scaler ONLY on train
scaler = MinMaxScaler()
scaler.fit(X_train)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

sequence_length = 10


# Train sequences
train_sequences = []

for user_id, group in train_df.groupby("user_id"):
    group = group.sort_values("timestamp")
    values = scaler.transform(group[feature_cols].fillna(0))

    for i in range(len(values) - sequence_length):
        train_sequences.append(values[i:i + sequence_length])

X_seq_train = np.array(train_sequences)


# Model
timesteps = X_seq_train.shape[1]
features = X_seq_train.shape[2]

inputs = Input(shape=(timesteps, features))

encoded = LSTM(64, activation="relu")(inputs)

decoded = RepeatVector(timesteps)(encoded)

decoded = LSTM(
    64,
    activation="relu",
    return_sequences=True
)(decoded)

outputs = TimeDistributed(Dense(features))(decoded)

lstm_model = Model(inputs, outputs)
lstm_model.compile(optimizer="adam", loss="mse")

print("\nTraining LSTM...")
lstm_model.fit(
    X_seq_train,
    X_seq_train,
    epochs=5,
    batch_size=64,
    verbose=1
)

# Save LSTM model
lstm_model.save("lstm_model.keras")


# Batch Prediction

print("\nPreparing test sequences...")

test_sequences = []
test_indices = []

for user_id, group in test_df.groupby("user_id"):
    group = group.sort_values("timestamp")
    values = scaler.transform(group[feature_cols].fillna(0))

    for i in range(len(values) - sequence_length):
        test_sequences.append(values[i:i + sequence_length])
        test_indices.append(group.index[i + sequence_length])

X_seq_test = np.array(test_sequences)

print(f"Total test sequences: {len(X_seq_test)}")


print("\nRunning batch prediction...")

X_pred = lstm_model.predict(
    X_seq_test,
    batch_size=256,
    verbose=1
)

errors = np.mean((X_seq_test - X_pred) ** 2, axis=(1, 2))


# Attach scores
test_df["lstm_score"] = np.nan
test_df.loc[test_indices, "lstm_score"] = errors


# Threshold
threshold = np.percentile(errors, 97)

# Save threshold for API
joblib.dump(threshold, "lstm_threshold.pkl")

test_df["lstm_anomaly"] = (
    test_df["lstm_score"] > threshold
).astype(int)


print("\nLSTM Results:")

valid_idx = test_df["lstm_score"].notna()

print(classification_report(
    test_df.loc[valid_idx, "is_anomaly"],
    test_df.loc[valid_idx, "lstm_anomaly"]
))


# Final decision
test_df["final_anomaly"] = (
    (test_df["if_anomaly"] == 1) |
    (test_df["lstm_anomaly"] == 1)
).astype(int)


def get_top_if_features(X_scaled_row: pd.DataFrame, top_n: int = 3) -> list[dict]:
    baseline = model.decision_function(X_scaled_row)[0]
    impacts = {}
    for col in feature_cols:
        X_perturbed = X_scaled_row.copy()
        X_perturbed[col] = 0.0
        perturbed_score = model.decision_function(X_perturbed)[0]
        impacts[col] = round(float(baseline - perturbed_score), 6)
    top = sorted(impacts.items(), key=lambda x: -abs(x[1]))[:top_n]
    return [{"feature": k, "impact": v} for k, v in top]


# Explainability 
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=feature_cols,
    index=test_df.index
)

def compute_reason(row):
    if row["final_anomaly"] != 1:
        return "normal behavior"
    top_features = get_top_if_features(
        X_test_scaled.loc[[row.name]]
    )
    return generate_reason(top_features)

test_df["reason"] = test_df.apply(compute_reason, axis=1)

test_df["alert_level"] = "low"

test_df.loc[
    (test_df["if_anomaly"] == 1) &
    (test_df["lstm_anomaly"] == 1),
    "alert_level"
] = "high"


# Save
test_df.to_csv("final_predictions.csv", index=False)

print("\nPipeline completed")
print(test_df.head())