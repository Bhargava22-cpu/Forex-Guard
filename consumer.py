import json
import os
from datetime import datetime
import requests
from kafka import KafkaConsumer

API_URL = os.getenv("API_URL", "http://localhost:8000/predict")
REQUEST_TIMEOUT = 5

consumer = KafkaConsumer(
    "forex-events",
    bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    auto_offset_reset="latest",
    enable_auto_commit=False,
    group_id="forexguard-consumer"
)

print("Listening for streaming events...\n")

for i, message in enumerate(consumer, start=1):
    event = message.value

    try:
        event["timestamp"] = str(datetime.fromisoformat(event["timestamp"]))

        response = requests.post(API_URL, json=event, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        result = response.json()

        print(f"event {i}")
        print(f"final_anomaly   : {result['final_anomaly']}")
        print(f"risk_score      : {result['risk_score']}")
        print(f"reason          : {result['reason']}")
        print()

        consumer.commit()

    except requests.exceptions.RequestException as e:
        print(f"event {i}: API request failed — {e}\n")

    except (KeyError, ValueError) as e:
        print(f"event {i}: bad event or response — {e}\n")