import numpy as np
import pandas as pd

WINDOW = 5
SESSION_GAP_SECONDS = 1800
HISTORY_CAP = 50  
feature_cols = [
    "event_type_encoded",
    "instrument_encoded",
    "time_diff",
    "unusual_time",
    "trade_zscore",
    "amount_zscore",
    "session_event_count",
    "recent_trade_count",
    "ip_changed",
    "ip_change_freq",
    "pnl_volatility",
    "instrument_freq",
    "withdraw_after_deposit"
]

event_type_map = {
    "login": 0,
    "deposit": 1,
    "withdrawal": 2,
    "trade": 3,
    "kyc_update": 4,
    "account_modification": 5
}

instrument_map = {
    "NONE": 0,
    "EURUSD": 1,
    "GBPUSD": 2,
    "USDJPY": 3,
    "AUDUSD": 4,
    "USDCAD": 5
}


def circular_hour_diff(h1, h2):
    diff = abs(h1 - h2)
    return min(diff, 24 - diff)


def compute_features(event, history, window=WINDOW):
    event_type = event["event_type"]
    instrument = event.get("instrument") or "NONE"

    event_type_encoded = event_type_map.get(event_type, 0)
    instrument_encoded = instrument_map.get(instrument, 0)

    # --- Time features ---
    if history:
        time_diff = (event["timestamp"] - history[-1]["timestamp"]).total_seconds()
    else:
        time_diff = 0.0

    past_hours = [h["timestamp"].hour for h in history]
    if past_hours:
        avg_hour = np.mean(past_hours)
        unusual_time = int(circular_hour_diff(event["timestamp"].hour, avg_hour) > 6)
    else:
        unusual_time = 0

    # --- Trade z-score: last `window` trade events only, current excluded from baseline ---
    past_trade_volumes = [
        h["trade_volume"] for h in history if h["event_type"] == "trade"
    ][-window:]

    if len(past_trade_volumes) >= 2:
        trade_mean = np.mean(past_trade_volumes)
        trade_std = np.std(past_trade_volumes)
        trade_zscore = (event.get("trade_volume", 0) - trade_mean) / (trade_std + 1e-5)
    else:
        trade_zscore = 0.0

    # --- Amount z-score: last `window` deposit/withdrawal events only ---
    past_amounts = [
        h["amount"] for h in history if h["event_type"] in ("deposit", "withdrawal")
    ][-window:]

    if len(past_amounts) >= 2:
        amount_mean = np.mean(past_amounts)
        amount_std = np.std(past_amounts)
        amount_zscore = (event.get("amount", 0) - amount_mean) / (amount_std + 1e-5)
    else:
        amount_zscore = 0.0

    # --- Session features (causal by construction — history is past-only) ---
    session_events = [
        h for h in history
        if (event["timestamp"] - h["timestamp"]).total_seconds() < SESSION_GAP_SECONDS
    ]
    session_event_count = len(session_events) + 1  # +1 for current event

    recent_trade_count = sum(
        1 for h in history[-window:] if h["event_type"] == "trade"
    )

    # --- IP features ---
    if history:
        ip_changed = int(event["ip_address"] != history[-1]["ip_address"])
    else:
        ip_changed = 0

    ip_change_flags = [
        int(history[i]["ip_address"] != history[i - 1]["ip_address"])
        for i in range(1, len(history))
    ]
    ip_change_flags.append(ip_changed)
    ip_change_freq = np.mean(ip_change_flags[-window:]) if ip_change_flags else 0.0

    # --- PnL volatility (trade events only) ---
    if event_type == "trade":
        past_pnls = [
            h["trade_volume"] * h["margin"]
            for h in history if h["event_type"] == "trade"
        ][-window:]
        current_pnl = event.get("trade_volume", 0) * event.get("margin", 0)
        pnl_window = past_pnls + [current_pnl]
        pnl_volatility = np.std(pnl_window) if len(pnl_window) >= 2 else 0.0
    else:
        pnl_volatility = 0.0

    # --- Instrument concentration (trade events only — fixes the
    if event_type == "trade":
        past_trade_instruments = [
            h["instrument"] for h in history if h["event_type"] == "trade"
        ][-window:]
        sequence = past_trade_instruments + [instrument]
        instrument_freq = 0
        for i in range(len(sequence) - 1, 0, -1):
            if sequence[i] == sequence[i - 1]:
                instrument_freq += 1
            else:
                break
    else:
        instrument_freq = 0

    # --- Withdraw after deposit ---
    recent_deposit = any(h["event_type"] == "deposit" for h in history[-window:])
    withdraw_after_deposit = int(event_type == "withdrawal" and recent_deposit)

    return {
        "event_type_encoded": event_type_encoded,
        "instrument_encoded": instrument_encoded,
        "time_diff": time_diff,
        "unusual_time": unusual_time,
        "trade_zscore": trade_zscore,
        "amount_zscore": amount_zscore,
        "session_event_count": session_event_count,
        "recent_trade_count": recent_trade_count,
        "ip_changed": ip_changed,
        "ip_change_freq": ip_change_freq,
        "pnl_volatility": pnl_volatility,
        "instrument_freq": instrument_freq,
        "withdraw_after_deposit": withdraw_after_deposit,
    }


def engineer_features_batch(df, window=WINDOW):
    """
    Batch driver. Calls compute_features() once per event, in
    chronological order per user, building up causal history as it
    goes. Slower than the old vectorized pandas version, but
    guarantees identical feature definitions to the real-time API.
    """
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    df["instrument"] = df["instrument"].fillna("NONE")
    df["trade_volume"] = df["trade_volume"].fillna(0)
    df["amount"] = df["amount"].fillna(0)
    df["margin"] = df["margin"].fillna(0)

    rows = df.to_dict("records")
    user_histories = {}
    feature_rows = []

    for row in rows:
        uid = row["user_id"]
        history = user_histories.setdefault(uid, [])

        feature_rows.append(compute_features(row, history, window=window))

        history.append(row)
        if len(history) > HISTORY_CAP:
            user_histories[uid] = history[-HISTORY_CAP:]

    feat_df = pd.DataFrame(feature_rows)
    return pd.concat([df.reset_index(drop=True), feat_df], axis=1)


if __name__ == "__main__":
    df = pd.read_csv("forex_events.csv", parse_dates=["timestamp"])
    df = engineer_features_batch(df)
    df.to_csv("engineered_features.csv", index=False)

    print("Feature engineering completed")
    print(df.head())