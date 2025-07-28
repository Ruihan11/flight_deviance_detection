# ✈️ Flight Deviance Detection

> Lightweight toolkit for spotting abnormal flight trajectories in real time.

---



### 1. Prerequisites

| Tool | Tested Version |
|------|----------------|
| **Python** | 3.10.12 (≥ 3.10 required) |
| **uv** | 0.1.34 or newer |

**Install uv** (one‑liner):  
> ```bash
> curl -Ls https://astral.sh/uv/install.sh | sh
> # or: pip install uv
> ```

### 2. Create & activate a virtual env

```bash
git clone https://github.com/Ruihan11/flight_deviance_detection.git
cd path-to-project
uv venv
source .venv/bin/activate  
uv sync
```
### 2.5 some modification before test run
Be ware that the training data comes from [data-processing](https://github.com/Ruihan11/flight_data_processing.git)
- plane_features_reduced.csv

### 3. Run
```bash
python3 main.py

python3 inference.py \
    --model models/RandomForestClassifier.pkl \
    --input data/plane_features_reduced.csv \
    --output outputs/rf_deviance_probability.csv
# python3 inference.py --help
```