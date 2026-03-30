# Explainability using top contributing features from Isolation Forest

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
}


def generate_reason(top_features: list[dict]) -> str:
    if not top_features:
        return "normal behaviour"

    labels = [
        feature_labels.get(f["feature"], f["feature"])
        for f in top_features
        if f["impact"] != 0 and f["feature"] in feature_labels
    ]

    return " + ".join(labels) if labels else "normal behaviour"