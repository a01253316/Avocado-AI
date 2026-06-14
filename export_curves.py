import json
import mlflow

TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(TRACKING_URI)

client = mlflow.tracking.MlflowClient()

runs = {
    "tiny":  "d2fa7fc23dd345dd98f539e10dad5752",
    "small": "2c181ae60d0c4853b0d352f0481f112b",
}

metrics_to_fetch = ["train_loss", "val_loss", "train_f1", "val_f1"]

result = {}
for preset, run_id in runs.items():
    result[preset] = {}
    for metric in metrics_to_fetch:
        history = client.get_metric_history(run_id, metric)
        result[preset][metric] = [
            {"step": m.step, "value": m.value} for m in history
        ]

print(json.dumps(result, indent=2))