from fraud_detection import main as run_detection
import argparse
import os

def detect_fraud(input_csv, output_csv=None, contamination=0.05, simulate_labels=False):
    # Create output path based on input filename
    if output_csv is None:
        base = os.path.basename(input_csv).replace(".csv", "_results.csv")
        output_csv = os.path.join("results", base)
        os.makedirs("results", exist_ok=True)

    args = argparse.Namespace(
        input=input_csv,
        output=output_csv,
        contamination=contamination,
        simulate_labels=simulate_labels
    )
    run_detection(args)
    return output_csv
