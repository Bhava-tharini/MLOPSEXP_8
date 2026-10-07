import json
import subprocess
from monitoring.drift_detection import detect_drift

result = detect_drift()
print(json.dumps(result, indent=2))

if result["drift_detected"]:
    print("Drift detected -> automated retraining started.")
    subprocess.run(["python", "-m", "src.preprocess"], check=True)
    subprocess.run(["python", "-m", "src.feature_selection"], check=True)
    subprocess.run(["python", "-m", "src.train"], check=True)
    print("Retraining completed.")
else:
    print("No significant drift -> model remains unchanged.")
