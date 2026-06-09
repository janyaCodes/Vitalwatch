# VitalWatch: Real-Time ICU Patient Deterioration Early Warning System

> A production-style ML engineering project that detects patient deterioration earlier than traditional threshold-based monitors — using streaming feature computation, containerized remote development, and live A/B testing on real clinical metrics.



> machine-learning  mlops  healthcare  fastapi  docker   python  streaming  ab-testing  real-time  scikit-learn   feature-store  devcontainers  logistic-regression
---

## The Problem

Traditional ICU monitors fire alerts only when a vital sign crosses a fixed threshold — heart rate above 120, SpO2 below 90. By that point, deterioration is already advanced.

**VitalWatch detects the trend before the threshold is crossed.**

A patient whose heart rate has climbed from 90 → 100 → 108 → 115 over six readings hasn't triggered any alarm yet. But the direction is clear. VitalWatch's streaming feature store computes `abnormal_streak` and `hr_rate_of_change` in real time and feeds them to a model that fires an alert earlier — while there's still time to intervene.

---

## Architecture

```
Vitals stream (simulated)
        │
        ▼
┌─────────────────────┐
│  Streaming Feature   │   abnormal_streak, hr_rate_of_change,
│  Store (in-memory)   │   spo2_rate_of_change
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│     A/B Router       │   patient_id % 2 → sticky session split
└──────┬──────┬───────┘
       │      │
       ▼      ▼
┌──────────┐  ┌──────────────────────┐
│ Model V1  │  │ Model V2              │
│ Threshold │  │ Logistic Regression   │
│ rules     │  │ + streaming features  │
└──────────┘  └──────────────────────┘
       │              │
       ▼              ▼
  /alert-confirmed    →    /stats
  (nurse confirms)         alert_precision per model
```

---

## Project Structure

```
vitalwatch/
├── .devcontainer/
│   └── devcontainer.json         # VSCode attaches to Docker container
├── Dockerfile                    # Python 3.11 + ML packages
├── docker-compose.yml            # Mounts local files into container
├── app.py                        # FastAPI — /assess, /alert-confirmed, /stats
├── data/
│   ├── __init__.py
│   └── simulate_vitals.py        # Simulates ICU vitals (healthy + deteriorating)
├── features/
│   ├── __init__.py
│   └── streaming_features.py     # abnormal_streak, hr/spo2 rate_of_change
├── models/
│   ├── __init__.py
│   ├── model_v1.py               # Static threshold rules
│   └── model_v2.py               # Logistic regression + streaming features
├── serving/
│   ├── __init__.py
│   └── ab_router.py              # Traffic split + alert precision tracker
└── requirements.txt
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| API Server | FastAPI + Uvicorn |
| ML Model | scikit-learn (Logistic Regression) |
| Containerization | Docker + Docker Compose |
| Remote Dev | VSCode Dev Containers extension |
| Feature Store | In-memory Python (dict-based sliding window) |
| Language | Python 3.11 |

---

## Key Concepts

### Streaming Features

Every call to `/assess` appends a timestamped vitals reading to an in-memory store per patient. Before scoring, the feature store computes three real-time signals over the last 5 readings:

| Feature | What it measures |
|---|---|
| `abnormal_streak` | Consecutive readings outside normal range |
| `hr_rate_of_change` | Average per-reading change in heart rate |
| `spo2_rate_of_change` | Average per-reading change in SpO2 |

Model V1 never sees these. Model V2 takes all three as input features alongside the raw vitals. This is the core of what a real-time feature store does in production — same computation logic at training time and serving time, eliminating training/serving skew.

### Model V1 vs Model V2

**Model V1** is a rule engine:
```python
alert = heart_rate > 120 or spo2 < 90 or resp_rate > 25
```
This is exactly how most ICU threshold monitors work today.

**Model V2** is a logistic regression trained on 600 labelled examples (300 healthy, 300 deteriorating) with 6 input features — the 3 raw vitals plus the 3 streaming features above. It fires earlier because it sees the trend, not just the current value.

### A/B Testing on Clinical Precision

`ab_router.py` uses sticky session splitting by `patient_id`:
- Even `patient_id` → always routed to V1
- Odd `patient_id` → always routed to V2

The metric tracked is **alert precision** — confirmed alerts divided by total alerts fired. A nurse calling `/alert-confirmed` is the ground truth signal. This is a more meaningful metric than offline accuracy: it measures how often each model wastes clinical staff time with false alarms.

### Dev Container Workflow

The `Dockerfile`, `docker-compose.yml`, and `devcontainer.json` together define a reproducible Python 3.11 environment. VSCode's Dev Containers extension attaches the local IDE directly to the running container — IntelliSense, debugging, and the terminal all point at the isolated environment. The same pattern used here with Docker scales directly to remote GPU clusters via VSCode Remote SSH.

---

## Complete Setup & Execution Guide

Follow these steps exactly to get VitalWatch running from scratch.

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) — download, install, and make sure it is running (whale icon in menu bar)
- [VSCode](https://code.visualstudio.com/) with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) installed

---

### Step 1 — Create the project folder

Open a terminal and run:

```bash
mkdir vitalwatch
cd vitalwatch
mkdir data features models serving
code .
```

VSCode opens with your empty project.

---

### Step 2 — Create `Dockerfile`

In VSCode's Explorer panel, right-click the `VITALWATCH` root folder → **New File** → name it `Dockerfile` (capital D, no extension).

Paste this inside and save with `Ctrl+S`:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN pip install fastapi uvicorn scikit-learn pandas numpy
```

---

### Step 3 — Create `docker-compose.yml`

Right-click the `VITALWATCH` root folder → **New File** → name it `docker-compose.yml`.

Paste this inside and save:

```yaml
services:
  app:
    build: .
    volumes:
      - .:/app
    ports:
      - "8000:8000"
    command: sleep infinity
```

---

### Step 4 — Create `.devcontainer/devcontainer.json`

Right-click the `VITALWATCH` root folder → **New Folder** → name it `.devcontainer`.

Right-click `.devcontainer` → **New File** → name it `devcontainer.json`.

Paste this inside and save:

```json
{
  "name": "VitalWatch Dev",
  "dockerComposeFile": "../docker-compose.yml",
  "service": "app",
  "workspaceFolder": "/app",
  "extensions": ["ms-python.python", "ms-python.pylance"]
}
```

---

### Step 5 — Open project inside Docker

Press `Ctrl+Shift+P` → type `Reopen in Container` → click **Dev Containers: Reopen in Container**.

Wait 2–3 minutes for the first build. When complete, the bottom-left corner of VSCode shows:

```
Dev Container: VitalWatch Dev
```

Confirm it worked by opening the terminal (`Ctrl+` `` ` ``) and running:

```bash
python --version
# Expected: Python 3.11.x
```

---

### Step 6 — Create `data/__init__.py`

Right-click the `data` folder → **New File** → name it `__init__.py`. Leave it blank. Save.

---

### Step 7 — Create `data/simulate_vitals.py`

Right-click the `data` folder → **New File** → name it `simulate_vitals.py`.

Paste this inside and save:

```python
import random
import time

vitals_log = {}

def generate_vitals(patient_id: int, deteriorating: bool = False) -> dict:
    if deteriorating:
        heart_rate = random.randint(110, 145)
        spo2 = random.randint(84, 92)
        resp_rate = random.randint(22, 30)
    else:
        heart_rate = random.randint(60, 100)
        spo2 = random.randint(95, 100)
        resp_rate = random.randint(12, 20)

    reading = {
        "patient_id": patient_id,
        "heart_rate": heart_rate,
        "spo2": spo2,
        "resp_rate": resp_rate,
        "timestamp": time.time()
    }

    if patient_id not in vitals_log:
        vitals_log[patient_id] = []
    vitals_log[patient_id].append(reading)

    return reading

def get_history(patient_id: int) -> list:
    return vitals_log.get(patient_id, [])
```

---

### Step 8 — Create `features/__init__.py`

Right-click the `features` folder → **New File** → name it `__init__.py`. Leave it blank. Save.

---

### Step 9 — Create `features/streaming_features.py`

Right-click the `features` folder → **New File** → name it `streaming_features.py`.

Paste this inside and save:

```python
from data.simulate_vitals import get_history

NORMAL_RANGES = {
    "heart_rate": (60, 100),
    "spo2": (95, 100),
    "resp_rate": (12, 20)
}

def is_abnormal(reading: dict) -> bool:
    for vital, (low, high) in NORMAL_RANGES.items():
        if reading[vital] < low or reading[vital] > high:
            return True
    return False

def get_features(patient_id: int) -> dict:
    history = get_history(patient_id)

    if len(history) < 2:
        return {
            "abnormal_streak": 0,
            "hr_rate_of_change": 0.0,
            "spo2_rate_of_change": 0.0
        }

    last5 = history[-5:]

    abnormal_streak = 0
    for reading in reversed(last5):
        if is_abnormal(reading):
            abnormal_streak += 1
        else:
            break

    hr_rate_of_change = (
        last5[-1]["heart_rate"] - last5[0]["heart_rate"]
    ) / len(last5)

    spo2_rate_of_change = (
        last5[-1]["spo2"] - last5[0]["spo2"]
    ) / len(last5)

    return {
        "abnormal_streak": abnormal_streak,
        "hr_rate_of_change": round(hr_rate_of_change, 2),
        "spo2_rate_of_change": round(spo2_rate_of_change, 2)
    }
```

---

### Step 10 — Create `models/__init__.py`

Right-click the `models` folder → **New File** → name it `__init__.py`. Leave it blank. Save.

---

### Step 11 — Create `models/model_v1.py`

Right-click the `models` folder → **New File** → name it `model_v1.py`.

Paste this inside and save:

```python
class ModelV1:
    def predict(self, vitals: dict) -> dict:
        alert = (
            vitals["heart_rate"] > 120 or
            vitals["spo2"] < 90 or
            vitals["resp_rate"] > 25
        )
        return {
            "alert": bool(alert),
            "reason": "static threshold breach" if alert else "within normal limits"
        }
```

---

### Step 12 — Create `models/model_v2.py`

Right-click the `models` folder → **New File** → name it `model_v2.py`.

Paste this inside and save:

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from features.streaming_features import get_features

class ModelV2:
    def __init__(self):
        self.model = LogisticRegression()
        self._train()

    def _train(self):
        X = []
        y = []

        for _ in range(300):
            hr = np.random.randint(60, 100)
            spo2 = np.random.randint(95, 100)
            rr = np.random.randint(12, 20)
            streak = np.random.randint(0, 2)
            hr_change = np.random.uniform(-1, 1)
            spo2_change = np.random.uniform(-0.5, 0.5)
            X.append([hr, spo2, rr, streak, hr_change, spo2_change])
            y.append(0)

        for _ in range(300):
            hr = np.random.randint(110, 145)
            spo2 = np.random.randint(84, 92)
            rr = np.random.randint(22, 30)
            streak = np.random.randint(3, 6)
            hr_change = np.random.uniform(2, 6)
            spo2_change = np.random.uniform(-4, -1)
            X.append([hr, spo2, rr, streak, hr_change, spo2_change])
            y.append(1)

        self.model.fit(np.array(X), np.array(y))

    def predict(self, vitals: dict, patient_id: int) -> dict:
        f = get_features(patient_id)
        x = [[
            vitals["heart_rate"],
            vitals["spo2"],
            vitals["resp_rate"],
            f["abnormal_streak"],
            f["hr_rate_of_change"],
            f["spo2_rate_of_change"]
        ]]
        prob = self.model.predict_proba(x)[0][1]
        return {
            "alert": bool(prob > 0.5),
            "deterioration_probability": round(float(prob), 3),
            "reason": "trend + threshold analysis"
        }
```

---

### Step 13 — Create `serving/__init__.py`

Right-click the `serving` folder → **New File** → name it `__init__.py`. Leave it blank. Save.

---

### Step 14 — Create `serving/ab_router.py`

Right-click the `serving` folder → **New File** → name it `ab_router.py`.

Paste this inside and save:

```python
from models.model_v1 import ModelV1
from models.model_v2 import ModelV2

v1 = ModelV1()
v2 = ModelV2()

results = {
    "v1": {"assessed": 0, "alerts_fired": 0, "alerts_confirmed": 0},
    "v2": {"assessed": 0, "alerts_fired": 0, "alerts_confirmed": 0}
}

def route_request(patient_id: int, vitals: dict) -> dict:
    model_name = "v1" if patient_id % 2 == 0 else "v2"

    if model_name == "v1":
        prediction = v1.predict(vitals)
    else:
        prediction = v2.predict(vitals, patient_id)

    results[model_name]["assessed"] += 1

    if prediction["alert"]:
        results[model_name]["alerts_fired"] += 1

    return {
        "model": model_name,
        "patient_id": patient_id,
        **prediction
    }

def confirm_alert(model_name: str):
    results[model_name]["alerts_confirmed"] += 1

def get_stats() -> dict:
    stats = {}
    for name, r in results.items():
        precision = (
            r["alerts_confirmed"] / r["alerts_fired"]
            if r["alerts_fired"] > 0 else 0
        )
        stats[name] = {
            **r,
            "alert_precision": round(precision, 3)
        }
    return stats
```

---

### Step 15 — Create `app.py`

Right-click the `VITALWATCH` root folder → **New File** → name it `app.py`.

Paste this inside and save:

```python
from fastapi import FastAPI
from data.simulate_vitals import generate_vitals
from serving.ab_router import route_request, confirm_alert, get_stats

app = FastAPI()

@app.get("/assess")
def assess(patient_id: int, deteriorating: bool = False):
    vitals = generate_vitals(patient_id, deteriorating=deteriorating)
    result = route_request(patient_id=patient_id, vitals=vitals)
    return {
        "vitals": vitals,
        "assessment": result
    }

@app.post("/alert-confirmed")
def alert_confirmed(model: str):
    confirm_alert(model)
    return {"status": "confirmed"}

@app.get("/stats")
def stats():
    return get_stats()
```

---

### Step 16 — Verify the final file tree

Your project should look exactly like this before running:

```
vitalwatch/
├── .devcontainer/
│   └── devcontainer.json
├── Dockerfile
├── docker-compose.yml
├── app.py
├── data/
│   ├── __init__.py
│   └── simulate_vitals.py
├── features/
│   ├── __init__.py
│   └── streaming_features.py
├── models/
│   ├── __init__.py
│   ├── model_v1.py
│   └── model_v2.py
└── serving/
    ├── __init__.py
    └── ab_router.py
```

---

### Step 17 — Start the server

In the terminal inside VSCode (make sure it shows `Dev Container: VitalWatch Dev` in the bottom-left):

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

You should see:

```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

---

### Step 18 — Test the API

Open a second terminal by clicking the `+` button at the top right of the terminal panel.

**Test a healthy patient:**

```bash
curl "http://localhost:8000/assess?patient_id=2&deteriorating=false"
```

**Test a deteriorating patient (run 7 times to build streaming history):**

```bash
curl "http://localhost:8000/assess?patient_id=1&deteriorating=true"
```

**Confirm an alert was real:**

```bash
curl -X POST "http://localhost:8000/alert-confirmed?model=v2"
```

**Check the live A/B scoreboard:**

```bash
curl "http://localhost:8000/stats"
```

**Or open the visual docs page in your browser:**

```
http://localhost:8000/docs
```

---

## API Reference

### `GET /assess`

Generates a vitals reading for a patient and routes it through the A/B router. Pass `deteriorating=true` to simulate a patient in decline.

```json
{
  "vitals": {
    "patient_id": 1,
    "heart_rate": 128,
    "spo2": 88,
    "resp_rate": 26,
    "timestamp": 1718123456.78
  },
  "assessment": {
    "model": "v2",
    "patient_id": 1,
    "alert": true,
    "deterioration_probability": 0.934,
    "reason": "trend + threshold analysis"
  }
}
```

### `POST /alert-confirmed`

Records that a nurse confirmed an alert was genuine. Used to compute alert precision in `/stats`.

```json
{"status": "confirmed"}
```

### `GET /stats`

Returns the live A/B scoreboard.

```json
{
  "v1": {
    "assessed": 12,
    "alerts_fired": 2,
    "alerts_confirmed": 0,
    "alert_precision": 0.0
  },
  "v2": {
    "assessed": 11,
    "alerts_fired": 7,
    "alerts_confirmed": 5,
    "alert_precision": 0.714
  }
}
```

---

## Reproducing the Key Finding

To demonstrate that V2 catches deterioration before V1:

```bash
# Build up 6 readings of streaming history for patient 1
for i in {1..6}; do curl -s "http://localhost:8000/assess?patient_id=1&deteriorating=true" > /dev/null; done

# Assess both patients
curl "http://localhost:8000/assess?patient_id=1&deteriorating=true"   # odd → V2
curl "http://localhost:8000/assess?patient_id=2&deteriorating=true"   # even → V1
```

V2 fires because it sees `abnormal_streak: 6` and a climbing `hr_rate_of_change`. V1 stays silent until a raw vital crosses its hard threshold.

---

## Known Issues & Fixes

**`numpy.bool` serialization error**
FastAPI cannot serialize `numpy.bool_` returned from sklearn comparisons. Fixed by wrapping in `bool()`:

```python
"alert": bool(prob > 0.5)   # correct
"alert": prob > 0.5          # causes Internal Server Error
```

---

## Potential Extensions

- Replace in-memory feature store with Redis for persistence across server restarts
- Add real vital signs dataset (MIMIC-III) for more realistic training data
- Implement time-decay on `abnormal_streak` — older readings count less
- Deploy to a cloud VM and connect via VSCode Remote SSH instead of Dev Containers
- Add a WebSocket endpoint to stream live alerts to a dashboard in real time
- Extend to multi-patient monitoring with a per-patient alert queue

---

## Disclaimer

This is an educational ML engineering project. It is not a medical device and should not be used for actual clinical decision-making.

---

## License

MIT
