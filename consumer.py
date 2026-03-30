import json
import os
from datetime import datetime

import requests
from kafka import KafkaConsumer

API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

# Connect to Kafka and subscribe to forex-events topic
consumer = KafkaConsumer(
    "forex-events",
    bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    auto_offset_reset="latest",
    enable_auto_commit=False,
    group_id=None
)
# group_id=None and enable_auto_commit=False so offsets are never saved.
# Every restart replays all events from the beginning — intentional for demo.
# In production, set group_id="forexguard-demo" and enable_auto_commit=True
# so the consumer resumes from where it left off after a restart.

print("Listening for streaming events...\n")

for i, message in enumerate(consumer, start=1):

    event = message.value
    event["timestamp"] = str(datetime.fromisoformat(event["timestamp"]))

    response = requests.post(API_URL, json=event)
    result = response.json()

    print(f"event {i}")
    print(f"final_anomaly   : {result['final_anomaly']}")
    print(f"risk_score      : {result['risk_score']}")
    print(f"reason          : {result['reason']}")
    print()