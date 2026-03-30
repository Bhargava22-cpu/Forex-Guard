import pandas as pd
import numpy as np

window = 5

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

# Fill missing values


def fill_missing_values(df):
    df["trade_volume"] = df["trade_volume"].fillna(0)
    df["lot_size"] = df["lot_size"].fillna(0)
    df["margin"] = df["margin"].fillna(0)
    df["amount"] = df["amount"].fillna(0)
    df["instrument"] = df["instrument"].fillna("NONE")

    return df


# Encode categorical features
def encode_features(df):
    df["event_type_encoded"] = df["event_type"].map(
        event_type_map).fillna(0).astype(int)
    df["instrument_encoded"] = df["instrument"].map(
        instrument_map).fillna(0).astype(int)

    return df


# Time Features
def add_time_features(df):
    df["time_diff"] = df.groupby("user_id")["timestamp"]\
        .diff().dt.total_seconds()

    df["time_diff"] = df["time_diff"].fillna(0)

    df["hour"] = df["timestamp"].dt.hour

    df["user_avg_hour"] = df.groupby("user_id")["hour"]\
        .transform("mean")

    df["unusual_time"] = (
        abs(df["hour"] - df["user_avg_hour"]) > 6
    ).astype(int)

    return df


# Trade Features
def add_trade_features(df):
    trade_mask = df["event_type"] == "trade"

    df["trade_mean"] = 0.0
    df["trade_std"] = 0.0

    df.loc[trade_mask, "trade_mean"] = (
        df[trade_mask]
        .groupby("user_id")["trade_volume"]
        .transform(lambda x: x.rolling(window, min_periods=1).mean())
    )

    df.loc[trade_mask, "trade_std"] = (
        df[trade_mask]
        .groupby("user_id")["trade_volume"]
        .transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
    )

    df["trade_mean"] = df.groupby("user_id")["trade_mean"]\
        .ffill().fillna(0)

    df["trade_std"] = df.groupby("user_id")["trade_std"]\
        .ffill().fillna(0)

    df["trade_zscore"] = (
        (df["trade_volume"] - df["trade_mean"]) /
        (df["trade_std"] + 1e-5)
    )

    return df


# Amount Features
def add_amount_features(df):
    money_mask = df["event_type"].isin(["deposit", "withdrawal"])

    df["amount_mean"] = 0.0
    df["amount_std"] = 0.0

    df.loc[money_mask, "amount_mean"] = (
        df[money_mask]
        .groupby("user_id")["amount"]
        .transform(lambda x: x.rolling(window, min_periods=1).mean())
    )

    df.loc[money_mask, "amount_std"] = (
        df[money_mask]
        .groupby("user_id")["amount"]
        .transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
    )

    df["amount_mean"] = df.groupby("user_id")["amount_mean"]\
        .ffill().fillna(0)

    df["amount_std"] = df.groupby("user_id")["amount_std"]\
        .ffill().fillna(0)

    df["amount_zscore"] = (
        (df["amount"] - df["amount_mean"]) /
        (df["amount_std"] + 1e-5)
    )

    return df


# Session Features
def add_session_features(df):
    df["new_session"] = (df["time_diff"] > 1800).astype(int)

    df["session_id"] = df.groupby("user_id")["new_session"].cumsum()

    df["session_event_count"] = (
        df.groupby(["user_id", "session_id"])["event_id"]
        .transform("count")
    )

    df["is_trade"] = (df["event_type"] == "trade").astype(int)

    df["recent_trade_count"] = (
        df.groupby("user_id")["is_trade"]
        .transform(lambda x: x.rolling(window, min_periods=1).sum())
    )

    return df


# IP Features
def add_ip_features(df):
    df["prev_ip"] = df.groupby("user_id")["ip_address"].shift(1)

    df["ip_changed"] = (df["ip_address"] != df["prev_ip"]).astype(int)
    df["ip_changed"] = df["ip_changed"].fillna(0)

    df["ip_change_freq"] = (
        df.groupby("user_id")["ip_changed"]
        .transform(lambda x: x.rolling(window, min_periods=1).mean())
    )

    return df


# PnL Volatility
def add_pnl_features(df):
    trade_mask = df["event_type"] == "trade"

    df["pnl_proxy"] = df["trade_volume"] * df["margin"]

    df["pnl_volatility"] = 0.0

    df.loc[trade_mask, "pnl_volatility"] = (
        df[trade_mask]
        .groupby("user_id")["pnl_proxy"]
        .transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
    )

    return df


# Instrument
def add_instrument_features(df):
    df["is_same_instrument"] = (df["instrument"] == df.groupby("user_id")[
                                "instrument"].shift(1)).astype(int)

    df["instrument_freq"] = (
        df.groupby("user_id")["is_same_instrument"]
        .transform(lambda x: x.rolling(window, min_periods=1).sum())
    )

    return df


# Deposit after Withdrawal
def add_withdrawal_features(df):
    deposit_flag = (df["event_type"] == "deposit").astype(int)
    withdraw_flag = (df["event_type"] == "withdrawal").astype(int)

    recent_deposits = deposit_flag.groupby(df["user_id"])\
        .transform(lambda x: x.rolling(5, min_periods=1).sum())

    df["withdraw_after_deposit"] = (
        (recent_deposits >= 1) &
        (withdraw_flag == 1)
    ).astype(int)

    return df


# Run all feature engineering steps
def engineer_features(df):
    df = fill_missing_values(df)
    df = encode_features(df)
    df = add_time_features(df)
    df = add_trade_features(df)
    df = add_amount_features(df)
    df = add_session_features(df)
    df = add_ip_features(df)
    df = add_pnl_features(df)
    df = add_instrument_features(df)
    df = add_withdrawal_features(df)

    return df

# Compute features for one event using recent history
def compute_features(data, history):

    event_type_encoded = event_type_map.get(data.event_type, 0)
    instrument_encoded = instrument_map.get(data.instrument, 0)

    # Time Features
    if history:
        previous_time = history[-1]["timestamp"]
        time_diff = (data.timestamp - previous_time).total_seconds()
    else:
        time_diff = 0

    previous_hours = [event["timestamp"].hour for event in history]

    if previous_hours:
        avg_hour = np.mean(previous_hours)
        unusual_time = int(abs(data.timestamp.hour - avg_hour) > 6)
    else:
        unusual_time = 0

    # Trade Features
    previous_trade_volumes = [
        event["trade_volume"]
        for event in history
        if event["trade_volume"] > 0
    ]

    if len(previous_trade_volumes) >= 2:
        trade_mean = np.mean(previous_trade_volumes)
        trade_std = np.std(previous_trade_volumes)
        trade_zscore = (
            data.trade_volume - trade_mean
        ) / (trade_std + 1e-5)
    else:
        trade_zscore = 0

    # Amount Features
    previous_amounts = [
        event["amount"]
        for event in history
        if event["amount"] > 0
    ]

    if len(previous_amounts) >= 2:
        amount_mean = np.mean(previous_amounts)
        amount_std = np.std(previous_amounts)
        amount_zscore = (
            data.amount - amount_mean
        ) / (amount_std + 1e-5)
    else:
        amount_zscore = 0

    # Session Features
    session_events = [
        event for event in history
        if (data.timestamp - event["timestamp"]).total_seconds() < 1800
    ]
    session_event_count = min(len(session_events) + 1, 10)

    recent_trade_count = sum(
        1 for event in history[-5:]
        if event["event_type"] == "trade"
    )

    # IP Features
    if history:
        ip_changed = int(data.ip_address != history[-1]["ip_address"])
    else:
        ip_changed = 0

    recent_ip_changes = []

    for i in range(1, len(history)):
        changed = int(
            history[i]["ip_address"] != history[i - 1]["ip_address"]
        )
        recent_ip_changes.append(changed)

    if history:
        recent_ip_changes.append(ip_changed)

    if recent_ip_changes:
        ip_change_freq = np.mean(recent_ip_changes[-5:])
    else:
        ip_change_freq = 0

    if data.event_type == "trade":
        pnl_values = [
            event["trade_volume"] * event["margin"]
            for event in history
            if event["trade_volume"] > 0
        ]
        current_pnl = data.trade_volume * data.margin
        pnl_values.append(current_pnl)
        if len(pnl_values) >= 2:
            pnl_volatility = np.std(pnl_values[-5:])
        else:
            pnl_volatility = 0
    else:
        pnl_volatility = 0

    recent_history = history[-5:]
    all_instruments = [e["instrument"]
                       for e in recent_history] + [data.instrument]
    instrument_freq = sum(
        1 for i in range(1, len(all_instruments))
        if all_instruments[i] == all_instruments[i - 1]
    )

    # Deposit after Withdrawal
    recent_deposit = any(
        event["event_type"] == "deposit"
        for event in history[-5:]
    )

    withdraw_after_deposit = int(
        data.event_type == "withdrawal" and recent_deposit
    )

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
        "withdraw_after_deposit": withdraw_after_deposit
    }


if __name__ == "__main__":

    # Load dataset
    df = pd.read_csv("forex_events.csv", parse_dates=["timestamp"])

    # Sort and fix timestamp
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # Run feature engineering
    df = engineer_features(df)

    # Save
    df.to_csv("engineered_features.csv", index=False)

    print("Feature engineering completed")
    print(df.head())
