import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

# Seed
seed = 42
np.random.seed(seed)
random.seed(seed)

# Config
num_users = 500
events_per_user = (70, 140)
anomaly_rate = 0.03

event_types = ["login", "deposit", "withdrawal",
               "trade", "kyc_update", "account_modification"]

instruments = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
devices = ["mobile", "desktop"]

user_types = ["normal", "active", "suspicious"]
user_type_probs = [0.7, 0.2, 0.1]


def generate_ip(base_ip):
    if random.random() < 0.9:
        return base_ip
    return f"{random.randint(10, 200)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 255)}"


def choose_event(user_type):
    if user_type == "normal":
        return random.choices(
            ["login", "trade", "deposit", "withdrawal"],
            weights=[0.4, 0.3, 0.2, 0.1]
        )[0]

    elif user_type == "active":
        return random.choices(
            ["login", "trade", "deposit"],
            weights=[0.3, 0.6, 0.1]
        )[0]

    else:
        return random.choices(
            ["login", "trade", "deposit", "withdrawal",
             "kyc_update", "account_modification"],
            weights=[0.2, 0.3, 0.1, 0.1, 0.15, 0.15]
        )[0]


events = []
event_id = 0
start_time = datetime(2026, 1, 1)


for user_id in range(num_users):

    user_type = np.random.choice(user_types, p=user_type_probs)

    base_ip = f"{random.randint(10, 200)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 255)}"
    base_trade = np.random.uniform(500, 2000)

    normal_start = random.randint(7, 10)
    normal_end = random.randint(20, 23)

    num_events = random.randint(*events_per_user)
    current_time = start_time + timedelta(days=random.randint(0, 30))

    for _ in range(num_events):

        event_id += 1
        event_type = choose_event(user_type)

        current_time += timedelta(minutes=random.randint(1, 180))
        current_time = current_time.replace(
            hour=random.randint(normal_start, normal_end))

        # Defaults
        amount = 0
        trade_volume = 0
        lot_size = 0
        instrument = None
        margin = 0
        session_duration = 0

        kyc_changed = 0
        account_modified = 0

        ip = generate_ip(base_ip)
        device = random.choice(devices)

        # Event Logic

        if event_type in ["deposit", "withdrawal"]:
            amount = round(max(100, np.random.normal(5000, 2000)), 2)

        elif event_type == "trade":
            trade_volume = round(
                max(0.1, np.random.normal(base_trade, 0.3 * base_trade)), 2)
            lot_size = round(np.random.uniform(0.1, 2), 3)
            margin = round(np.random.uniform(100, 1000), 2)
            instrument = random.choice(instruments)

        elif event_type == "login":
            session_duration = random.randint(1, 120)

        elif event_type == "kyc_update":
            kyc_changed = 1

        elif event_type == "account_modification":
            account_modified = 1

        # Anomaly Injection

        is_anomaly = 0

        if random.random() < anomaly_rate:
            is_anomaly = 1

            anomaly_type = random.choice(
                ["amount", "trade", "time", "ip", "behavior"]
            )

            if anomaly_type == "amount" and event_type in ["deposit", "withdrawal"]:
                amount = round(np.random.uniform(50000, 200000), 2)

            elif anomaly_type == "trade" and event_type == "trade":
                trade_volume = round(np.random.uniform(
                    10 * base_trade, 20 * base_trade), 2)
                margin = round(np.random.uniform(5000, 20000), 2)

            elif anomaly_type == "time":
                off_hours = [h for h in range(
                    0, 24) if h < normal_start - 2 or h > normal_end + 2]
                current_time = current_time.replace(
                    hour=random.choice(off_hours))

            elif anomaly_type == "ip":
                ip = f"172.16.{random.randint(0, 255)}.{random.randint(1, 255)}"

            elif anomaly_type == "behavior":
                if event_type == "withdrawal":
                    kyc_changed = 1
                if event_type == "login":
                    ip = f"172.16.{random.randint(0, 255)}.{random.randint(1, 255)}"

        # Store

        events.append({
            "event_id": event_id,
            "user_id": user_id,
            "timestamp": current_time,
            "event_type": event_type,
            "ip_address": ip,
            "device": device,
            "session_duration": session_duration,
            "amount": amount,
            "trade_volume": trade_volume,
            "lot_size": lot_size,
            "instrument": instrument,
            "margin": margin,
            "kyc_changed": kyc_changed,
            "account_modified": account_modified,
            "user_type": user_type,
            "is_anomaly": is_anomaly
        })


# Save

df = pd.DataFrame(events)

df = df.sort_values("timestamp").reset_index(drop=True)

df.to_csv("forex_events.csv", index=False)

print(df.head())
print("Total events:", len(df))
