# scripts/generate_test_csv.py
import pandas as pd
import numpy as np
import os
from datetime import datetime

def generate():
    rows = []

    for i in range(50):
        choice = np.random.choice(["wafer","gas","temperature","soil","light"])

        if choice == "wafer":
            row = {"wafer_id": f"W{i}", "timestamp": int(datetime.now().timestamp()*1000)}
            for s in range(1,31):
                row[f"sensor_{s}"] = np.random.randn()
            row["faulty"] = ""

        elif choice == "gas":
            row = {
                "timestramp_millies": int(datetime.now().timestamp() * 1000),
                "mq2_value": np.random.rand()*5,
                "temperature": 20 + np.random.rand()*10,
                "humidity": 30 + np.random.rand()*20,
                "label": ""
            }

        elif choice == "temperature":
            row = {
                "timestramp": int(datetime.now().timestamp() * 1000),
                "sensor_value": 15 + np.random.rand()*10,
                "label": ""
            }

        elif choice == "soil":
            row = {
                "timestamp_ms": int(datetime.now().timestamp() * 1000),
                "sensor_value": np.random.rand()*1000,
                "label": ""
            }

        else:  # light
            row = {
                "timestamp": int(datetime.now().timestamp() * 1000),
                "ldr_value": np.random.rand() * 1000,
                "voltage": np.random.rand(),
                "resistance": np.random.rand()*100,
                "ambient_light": np.random.rand()*200,
                "status": ""
            }

        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/test_input_50.csv", index=False)
    print("Generated data/test_input_50.csv")

if __name__ == "__main__":
    generate()
