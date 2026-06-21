# ForexGuard
**Real-Time Trader Anomaly Detection Engine**

---

## 1. Overview

ForexGuard is a real-time anomaly detection system for a forex brokerage environment. It monitors user behavior across the client portal and trading terminal, detects suspicious patterns using machine learning, and generates human-readable risk alerts for a compliance team.

The system is designed as a streaming pipeline — each incoming event is processed individually against the user's recent history, and an anomaly score with explanation is returned instantly via a REST API

---

## 2. System Architecture

| Layer | Description |
|---|---|
| Data generation | ~50,000 synthetic events across 500 users spread across January 2026. Each user has an assigned active hour window and a real per-session clustering pattern. Normal events cluster within that window; anomalies deliberately fall outside it. |
| Feature engineering | Raw events converted into behavioral signals — rolling z-scores, time deviation from personal baseline, IP deviation, PnL volatility, session metrics, instrument concentration. A single shared function computes these features identically for both training and live inference. |
| Modeling | Isolation Forest for point anomalies. LSTM Autoencoder for sequence anomalies, trained only on anomaly-free windows. Both trained on engineered features. |
| Streaming pipeline | Events streamed via Kafka. Consumer forwards each event to the API. Features computed live against per-user history maintained in fakeredis. |
| API layer | FastAPI endpoint accepting event JSON, returning anomaly label, risk score, and human-readable explanation. |

**Pipeline flow:**
```
Producer → Kafka → Consumer → POST /predict → API
→ Update user history → Compute features (shared function) → Isolation Forest (raw features)
→ LSTM Autoencoder (if 10+ events, scaled features) → Risk score + reason → Response
```

---

## 3. Repository Structure

```
forexguard/
├── dataset.py                # Synthetic data generation
├── feature_engineering.py    # Shared feature logic — batch + real-time use the same function
├── models.py                 # Isolation Forest + LSTM training
├── reason.py                 # Explainability - human-readable alerts
├── api.py                    # FastAPI prediction endpoint
├── consumer.py               # Kafka consumer — forwards events to API
├── producer.py                # Kafka producer — streams synthesised events
├── forex_events.csv          # Generated raw dataset
├── engineered_features.csv   # Feature-engineered dataset
├── final_predictions.csv     # Model output on test set
├── if_model.pkl              # Saved Isolation Forest
├── lstm_model.keras          # Saved LSTM Autoencoder
├── scaler.pkl                # Saved MinMaxScaler (fit on normal-only data)
└── lstm_threshold.pkl        # Saved anomaly threshold (from clean reconstruction errors)
```

---

## 4. Setup Instructions

### 4.1 Install dependencies
```bash
pip install fastapi uvicorn pandas numpy scikit-learn tensorflow joblib fakeredis kafka-python requests
```

### 4.2 Generate dataset
```bash
python dataset.py
```
Produces `forex_events.csv` with ~50,000 events. Seed fixed at 42 for reproducibility.

### 4.3 Run feature engineering
```bash
python feature_engineering.py
```
Produces `engineered_features.csv`, computed via the same `compute_features()` function used by the live API — no separate batch implementation, so there is no train/serve feature skew.

### 4.4 Train models
```bash
python models.py
```
Trains both models, saves all artifacts, and prints classification reports.

**Note:** if you're on macOS or hit an unkillable hang during `lstm_model.fit()`, this is a known TensorFlow thread-pool deadlock on some environments. Add the following at the very top of `models.py`, before any other imports:
```python
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
```

### 4.5 Run the streaming demo

**First time only — Create Kafka container**
```bash
docker run -d --name kafka -p 9092:9092 \
  -e KAFKA_NODE_ID=1 \
  -e KAFKA_PROCESS_ROLES=broker,controller \
  -e KAFKA_CONTROLLER_QUORUM_VOTERS=1@localhost:9093 \
  -e KAFKA_LISTENERS=PLAINTEXT://:9092,CONTROLLER://:9093 \
  -e KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092 \
  -e KAFKA_LISTENER_SECURITY_PROTOCOL_MAP=PLAINTEXT:PLAINTEXT,CONTROLLER:PLAINTEXT \
  -e KAFKA_CONTROLLER_LISTENER_NAMES=CONTROLLER \
  -e KAFKA_INTER_BROKER_LISTENER_NAME=PLAINTEXT \
  -e KAFKA_AUTO_CREATE_TOPICS_ENABLE=true \
  -e CLUSTER_ID=MkU3OEVBNTcwNTJENDM2Qg== \
  confluentinc/cp-kafka:7.6.0
```

**Terminal 1 — Start Kafka**
```bash
docker start kafka
```
Give it 10–30 seconds after starting before connecting — check readiness with `docker logs kafka --tail 50` (look for the broker startup line).

**Terminal 2 — Start API**
```bash
uvicorn api:app --reload
```
Runs at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

**Terminal 3 — Start Consumer**
```bash
python consumer.py
```

**Terminal 4 — Run Producer**
```bash
python producer.py
```

Watch Terminal 3 for live anomaly results.

---

## 5. Dataset

### 5.1 Overview
Fully synthetic dataset with ~50,000 events across 500 users, spread across January 2026. Each user is assigned a personal peak active hour, and events are generated in sessions clustered around that hour (rather than each event independently re-rolling its timestamp across the full active window). Sessions are spaced hours apart so date and session boundaries progress naturally across the month.

### 5.2 Event types

| Event type | Source | Key fields |
|---|---|---|
| login | Client portal | ip_address, device, session_duration |
| deposit | Client portal | amount |
| withdrawal | Client portal | amount |
| trade | Trading terminal | trade_volume, lot_size, margin, instrument |
| kyc_update | Client portal | kyc_changed flag |
| account_modification | Client portal | account_modified flag |

### 5.3 User behavior profiles
- **Normal (70%)** — 1 to 2 logins per day, small consistent trade volumes, stable IP, occasional deposits
- **Active (20%)** — frequent trades, multiple instruments, higher activity density
- **Suspicious (10%)** — irregular timing, large transactions, IP switching, KYC changes before withdrawals

### 5.4 Injected anomalies
Anomalies are injected at a 3% rate across five types. The anomaly type sampled for a given event is restricted to types that are actually applicable to that event's `event_type`, so every injected anomaly is guaranteed to modify the record — there are no mislabeled no-op anomalies.

- **Amount anomaly** — deposit or withdrawal 10 to 40x above normal (50,000 to 200,000)
- **Trade anomaly** — trade volume 10 to 20x the user's baseline, with margin derived from an abnormally high leverage ratio (thin margin relative to trade size, not just two unrelated large numbers)
- **Time anomaly** — activity forced to a random hour more than 2 hours outside the user's personal active window
- **IP anomaly** — login from a 172.16.x.x address, distinct from the user's established base IP
- **Behavior anomaly** — KYC update immediately preceding a withdrawal, or IP switch on login

### 5.5 Trade realism
`margin` is no longer sampled independently of `trade_volume`. Each user has a fixed personal leverage ratio, and normal-trade margin is derived from `trade_volume / leverage`. This means `pnl_volatility` (`trade_volume × margin`) reflects a real coupled risk relationship, and the trade anomaly type represents genuinely thin-margin, over-leveraged behavior rather than two independently large values.

---

## 6. Feature Engineering

Feature engineering converts raw events into behavioral signals by comparing each event against the user's established baseline, using only information available **before** the current event — no feature is computed using future events relative to the one being scored.

A single function, `compute_features()`, is the only implementation of this logic in the codebase. It is called once per event, in chronological order, by both the batch training script and the live API. This eliminates train/serve skew: there is exactly one definition of each feature, used identically everywhere.

| Feature | Type | Description |
|---|---|---|
| time_diff | Time | Seconds since user's last event |
| unusual_time | Time | 1 if current hour deviates more than 6 hours (with midnight wraparound handled) from the user's average hour over **past** events only |
| trade_zscore | Trade | Z-score of current trade volume vs the user's last 5 trade events (current event excluded from its own baseline) |
| amount_zscore | Financial | Z-score of current deposit or withdrawal vs the user's last 5 deposit/withdrawal events |
| session_event_count | Session | Count of past events within the current 30-minute window, plus the current event |
| recent_trade_count | Session | Number of trades in the last 5 events |
| ip_changed | IP | 1 if IP address differs from the previous event |
| ip_change_freq | IP | Rolling mean of ip_changed over the last 5 events |
| pnl_volatility | Trading | Rolling std of trade_volume × margin over the user's last 5 trades — measures erratic, leverage-aware risk exposure |
| instrument_freq | Trading | Length of the current consecutive same-instrument streak, computed over trade events only |
| withdraw_after_deposit | Financial | 1 if current event is a withdrawal and a deposit occurred in the last 5 events |
| event_type_encoded | Categorical | Fixed integer encoding of event type using a deterministic map |
| instrument_encoded | Categorical | Fixed integer encoding of instrument using a deterministic map |

Categorical encoding uses fixed hardcoded maps rather than `pandas cat.codes`, ensuring consistent encoding between training and inference regardless of which categories appear in a given run.

---

## 7. Models

### 7.1 Isolation Forest
Isolation Forest isolates observations by randomly selecting a feature and a split value. Anomalous points require fewer splits to isolate and receive lower anomaly scores. It operates on individual events and requires no sequence context. It is trained and scored on **raw, unscaled** features throughout the pipeline (training, batch evaluation, explainability, and live API).

- 100 estimators, contamination = 0.03
- Input: 13 engineered features per event
- Output: `decision_function` score normalised to a 0–1 risk score
- Chosen because it handles high-dimensional tabular data without labeled training data and is fast enough for real-time inference

### 7.2 LSTM Autoencoder
The LSTM Autoencoder learns to reconstruct sequences of normal user behavior. At inference time, sequences that deviate from learned patterns produce high reconstruction errors, flagged as anomalies.

- Architecture: LSTM encoder (64 units) → RepeatVector → LSTM decoder (64 units) → Dense output
- Sequence length: 10 consecutive events per user
- **Training data is filtered to exclude any 10-event window containing an anomalous event** — the autoencoder only ever learns to reconstruct genuinely normal behavior
- Features are scaled via MinMaxScaler **fit on normal-only training rows** before being passed to the LSTM
- Threshold: 97th percentile of reconstruction error on the clean training set, saved as `lstm_threshold.pkl`
- Requires at least 10 events before activating — new users are evaluated by Isolation Forest only until history builds

**Known limitation under current evaluation:** reconstruction error is currently averaged across the full 10-timestep window, then attributed to the most recent event in that window. Because one anomalous event can elevate the reconstruction error of several overlapping windows, this currently produces a high false-positive rate relative to Isolation Forest on this dataset. A last-timestep-only error calculation was identified as a likely fix but has not yet been implemented. Given that most injected anomaly types in this dataset are point-detectable, Isolation Forest is currently the stronger and more reliable of the two models; whether LSTM's sequence-only contribution is worth its added complexity has not yet been empirically validated (see Limitations).

### 7.3 Final decision

| Property | Isolation Forest | LSTM Autoencoder |
|---|---|---|
| Anomaly type | Point anomalies | Sequence anomalies |
| Context window | Single event | 10 consecutive events |
| Min history needed | None | 10 events |
| Feature scale | Raw | MinMax scaled (normal-only fit) |
| Output used | Risk score + anomaly flag | Anomaly flag only |

Both models run independently. The final anomaly flag uses OR logic — if either model flags an event it is marked anomalous. If both flag it, the alert level is elevated to `"high"`. The risk score is derived entirely from the Isolation Forest decision function; the LSTM contributes a binary flag only.

**Known trade-off, accepted as-is:** because LSTM never activates before 10 events, `alert_level` can only reach `"high"` for users with at least 10 events of history — a severe anomaly on a brand-new account will currently always show as `"low"` severity. This is a deliberate, documented limitation rather than an oversight.

---

## 8. Real-Time Processing

Training is batch-based on the full dataset. Streaming simulation applies to inference only — the API processes each event sequentially, maintaining per-user history in fakeredis and computing all features live against that history using the same `compute_features()` function used in training.

Per-event processing steps:
- Producer streams event into Kafka
- Consumer receives event and forwards it to `POST /predict` (with retry-safe error handling — a single bad event no longer halts the consumer)
- API retrieves stored history for this `user_id` from fakeredis
- Compute all 13 features using `compute_features()` against that history
- Run Isolation Forest on the raw feature vector
- If 10 or more engineered-feature events exist in history, scale the sequence and run the LSTM
- Compute top contributing features from Isolation Forest, on raw features
- Append current event to user history, capped at the last 10 events
- Return `final_anomaly`, `risk_score`, and `reason`

---

## 9. API Reference

### `GET /`
Health check.
```json
{"message": "API is running"}
```

### `POST /predict`

**Request:**
```json
{
  "user_id": 1,
  "timestamp": "2026-03-15T10:30:00",
  "event_type": "trade",
  "trade_volume": 1200.0,
  "amount": 0,
  "margin": 500.0,
  "instrument": "EURUSD",
  "ip_address": "45.123.12.89"
}
```

**Response:**
```json
{
  "final_anomaly": 1,
  "risk_score": 0.4731,
  "reason": "ip change + high trade spike"
}
```

For a meaningful prediction, send a sequence of normal events for the same `user_id` first to build a behavioral baseline, then send the suspicious event. A single isolated event will score low because `compute_features()` has no history to compare against.

---

## 10. Explainability

Every prediction includes a `reason` string derived from the top contributing features identified by the Isolation Forest perturbation method, computed on **raw (unscaled) features** — matching the scale the model was actually trained on, so a perturbed feature of `0.0` is genuinely neutral rather than an artifact of MinMax scaling.

For each feature, the model score is recomputed with that feature zeroed out — the features whose removal most reduces the anomaly score are ranked as top contributors and mapped to plain-English labels.

| Reason | Feature |
|---|---|
| high trade spike | trade_zscore |
| unusual transaction | amount_zscore |
| withdrawal after recent deposit | withdraw_after_deposit |
| ip change | ip_changed |
| frequent ip switching | ip_change_freq |
| activity at unusual time | unusual_time |
| trade burst | recent_trade_count |
| high session activity | session_event_count |
| high pnl volatility | pnl_volatility |
| instrument concentration | instrument_freq |

**Known gap:** `event_type_encoded`, `instrument_encoded`, and `time_diff` do not currently have entries in this label map. If one of these ranks as a top contributor, it is silently dropped from the reason string — in the edge case where it's the only non-trivial contributor, this can produce `"normal behaviour"` for an event that was in fact flagged. Not yet fixed.

---

## 11. Assumptions, Trade-offs, and Limitations

### Assumptions
- User behavior is consistent enough within the simulation period that rolling statistics over 5 events of the relevant type capture a meaningful personal baseline
- The 3% anomaly rate approximates real-world fraud prevalence in a retail forex context — note this also matches the Isolation Forest `contamination` parameter exactly, which is a leaked-ground-truth simplification rather than something a real deployment could know in advance
- `trade_volume × margin`, with margin now derived from a per-user leverage ratio, is a more realistic proxy for PnL exposure than independently-sampled margin, but is still an approximation in the absence of live market price data
- A 30-minute gap between events is a reasonable session boundary
- Each user's active hour window and leverage ratio remain stable over the simulation period — behavioral drift over time is not modeled

### Trade-offs
- Isolation Forest operates on single events (with some indirect sequence awareness via rolling-window features) and cannot detect anomalies that only emerge across a longer sequence — this is the intended justification for including the LSTM, though it has not yet been empirically confirmed that any injected anomaly type in this dataset actually requires sequence-level detection
- LSTM requires at least 10 events before it activates — new users are evaluated by Isolation Forest only during the cold-start period, and cannot reach `"high"` alert severity until then (an accepted, documented limitation)
- The risk score is derived from Isolation Forest only; the LSTM contributes a binary anomaly flag, not a magnitude
- The final anomaly decision uses OR logic between the two models, which compounds their individual false-positive rates rather than averaging them — accepted on the assumption that missing a real anomaly is costlier than an extra false alarm for a compliance reviewer to dismiss
- Training is batch-based on the full dataset. True online learning is not supported by Isolation Forest or the LSTM Autoencoder in their standard implementations
- The LSTM threshold is fixed at the 97th percentile of clean training reconstruction errors and does not adapt over time
- LSTM reconstruction error is currently computed across the full 10-event window rather than the most recent event only, which appears to meaningfully inflate its false-positive rate relative to Isolation Forest on this dataset (see Section 7.2)

### Limitations
- User history resets on every server restart since fakeredis is in-memory, and is not shared across multiple API worker processes — both are accepted limitations for the current scope; swap for real Redis in production
- The API has no authentication
- Kafka is used in a 1:1, single-consumer, non-batched configuration that does not yet exercise partitioning, parallel consumer groups, or backpressure handling — it currently demonstrates the streaming pattern more than it solves a load problem at this scale
- Alerts are returned as API responses only — they are not pushed anywhere. A real compliance team would receive them via a message queue or dashboard
- The model is trained once and never updated. If user behavior shifts over time the model becomes stale and will need retraining
- `instrument_freq` only catches consecutive same-instrument repetition. A user who alternates between two instruments would not be flagged even if they never diversify
- Coordinated activity across multiple accounts — such as mirror trades or shared IPs — is not detected since the system looks at each user independently
- PnL volatility is approximated using trade volume and margin since actual profit and loss requires live market price data
- Three engineered features (`event_type_encoded`, `instrument_encoded`, `time_diff`) lack explanation labels in `reason.py` and are silently excluded from the `reason` string if selected as top contributors
- The synthetic anomaly generation and the engineered features are closely coupled by construction (e.g., amount anomalies are designed to be caught by `amount_zscore`), so reported precision/recall on this dataset likely overstates performance on real, less-structured fraud patterns

---

## 12. Path to Production
- Replace fakeredis with real Redis so user history persists across restarts and scales across multiple servers
- Add API authentication so only authorised systems can call `/predict`
- Publish high-risk alerts to Kafka or RabbitMQ instead of only returning them in the API response
- Add monitoring to track prediction latency and anomaly rate over time, and trigger model retraining when drift is detected
- Make Kafka usage genuinely load-bearing: partition by `user_id`, batch consumer reads, run parallel consumers
- Fix LSTM reconstruction error to score the most recent event in a window rather than the window average, and re-evaluate whether the LSTM materially improves detection over Isolation Forest alone on this dataset
- Add labels for the three unmapped features in `reason.py`, or restrict explainability candidate features to only those with labels
- Reconsider cold-start alert severity so a high-confidence anomaly from a new user isn't always capped at `"low"`

---

## 13. Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| Data | Pandas, NumPy | Dataset generation and feature engineering |
| ML — baseline | scikit-learn | Isolation Forest |
| ML — advanced | TensorFlow / Keras | LSTM Autoencoder |
| Scaling | scikit-learn | MinMaxScaler feature normalization (fit on normal-only data) |
| Serialization | joblib | Model, scaler, threshold persistence |
| API | FastAPI + Pydantic | REST endpoint with typed input validation |
| Server | Uvicorn | ASGI server |
| Streaming | Kafka + kafka-python | Event streaming pipeline |
| State | fakeredis → Redis | Per-user history persistence |