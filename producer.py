import json
import os
import time

from kafka import KafkaProducer

KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

# Connect to Kafka
producer = KafkaProducer(
    bootstrap_servers=KAFKA_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

# Synthesised 10-event scenario for user 99
# Events 1-9: normal trading behaviour, consistent IP, normal volumes
# Event 10:    anomaly — trade spike 20x normal + IP change
events = [
    {"user_id": 99, "timestamp": "2026-01-10T09:00:00", "event_type": "login",   "trade_volume": 0,       "amount": 0,      "margin": 0,      "instrument": "NONE",   "ip_address": "45.123.12.89"},
    {"user_id": 99, "timestamp": "2026-01-10T09:20:00", "event_type": "trade",   "trade_volume": 1200.0,  "amount": 0,      "margin": 450.0,  "instrument": "EURUSD", "ip_address": "45.123.12.89"},
    {"user_id": 99, "timestamp": "2026-01-10T09:50:00", "event_type": "trade",   "trade_volume": 1100.0,  "amount": 0,      "margin": 420.0,  "instrument": "EURUSD", "ip_address": "45.123.12.89"},
    {"user_id": 99, "timestamp": "2026-01-10T10:20:00", "event_type": "trade",   "trade_volume": 1300.0,  "amount": 0,      "margin": 480.0,  "instrument": "GBPUSD", "ip_address": "45.123.12.89"},
    {"user_id": 99, "timestamp": "2026-01-10T10:50:00", "event_type": "login",   "trade_volume": 0,       "amount": 0,      "margin": 0,      "instrument": "NONE",   "ip_address": "45.123.12.89"},
    {"user_id": 99, "timestamp": "2026-01-10T11:10:00", "event_type": "trade",   "trade_volume": 1150.0,  "amount": 0,      "margin": 460.0,  "instrument": "EURUSD", "ip_address": "45.123.12.89"},
    {"user_id": 99, "timestamp": "2026-01-10T11:40:00", "event_type": "trade",   "trade_volume": 1250.0,  "amount": 0,      "margin": 470.0,  "instrument": "USDJPY", "ip_address": "45.123.12.89"},
    {"user_id": 99, "timestamp": "2026-01-10T12:10:00", "event_type": "login",   "trade_volume": 0,       "amount": 0,      "margin": 0,      "instrument": "NONE",   "ip_address": "45.123.12.89"},
    {"user_id": 99, "timestamp": "2026-01-10T12:30:00", "event_type": "trade",   "trade_volume": 1180.0,  "amount": 0,      "margin": 440.0,  "instrument": "EURUSD", "ip_address": "45.123.12.89"},
    {"user_id": 99, "timestamp": "2026-01-10T13:45:00", "event_type": "trade",   "trade_volume": 24000.0, "amount": 0,      "margin": 9000.0, "instrument": "EURUSD", "ip_address": "172.16.99.254"},
]

print(f"Streaming {len(events)} events for user 99\n")

# Stream events one per second
for i, event in enumerate(events, start=1):
    producer.send("forex-events", event)
    print(f"sent event {i}")
    time.sleep(1)

producer.flush()
print("\nFinished streaming events")