feature_labels = {
    "trade_zscore":           "high trade spike",
    "amount_zscore":          "unusual transaction",
    "withdraw_after_deposit": "withdrawal after recent deposit",
    "ip_changed":             "ip change",
    "ip_change_freq":         "frequent ip switching",
    "unusual_time":           "activity at unusual time",
    "recent_trade_count":     "trade burst",
    "session_event_count":    "high session activity",
    "pnl_volatility":         "high pnl volatility",
    "instrument_freq":        "instrument concentration",
    "event_type_encoded":     "unusual event type pattern",
    "time_diff":              "irregular timing between actions",
    "instrument_encoded":     "unusual instrument",
}

IMPACT_THRESHOLD = 1e-4

def generate_reason(top_features: list[dict]) -> str:
    if not top_features:
        return "normal behaviour"
    labels = [
        feature_labels.get(f["feature"], f["feature"])
        for f in top_features
        if abs(f["impact"]) > IMPACT_THRESHOLD
    ]
    return " + ".join(labels) if labels else "normal behaviour"