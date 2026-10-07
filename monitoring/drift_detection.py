import json
import pandas as pd
from scipy.stats import ks_2samp
from src.config import REFERENCE_PATH, CURRENT_PATH

FEATURES = ["N","P","K","temperature","humidity","ph","rainfall"]

def detect_drift(threshold=0.05):
    ref = pd.read_csv(REFERENCE_PATH)
    cur = pd.read_csv(CURRENT_PATH)
    results = {}
    drifted = False
    for f in FEATURES:
        stat, p = ks_2samp(ref[f], cur[f])
        results[f] = {"ks_statistic": float(stat), "p_value": float(p), "drift": bool(p < threshold)}
        drifted |= p < threshold
    return {"drift_detected": bool(drifted), "features": results}

if __name__ == "__main__":
    result = detect_drift()
    print(json.dumps(result, indent=2))
