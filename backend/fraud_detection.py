"""
fraud_detection.py

"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

plt.style.use("seaborn-v0_8")

# ---------- Helpers ----------
def count_hub_scans(x):
    if pd.isna(x):
        return 0
    s = str(x)
    for sep in ['|', ';', '>', '->', ',']:
        if sep in s:
            return len([p for p in s.split(sep) if p.strip() != ''])
    return len([p for p in s.split() if p.strip() != ''])

# ---------- Main ----------
def main(args):
    inp = args.input
    out = args.output
    contamination = args.contamination
    simulate_labels = args.simulate_labels
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)

    print(f"Loading data from: {inp}")
    df = pd.read_csv(inp, low_memory=False)
    print("Columns detected:", list(df.columns))
    print("Rows:", len(df))
    print()

    # ---------- Explicit mapping for Delhivery dataset ----------
    pickup_col = "od_start_time"
    delivery_col = "od_end_time"
    tracking_col = "trip_uuid"
    source_col = "source_center"
    dest_col = "destination_center"

    # Convert pickup & delivery timestamps
    if pickup_col in df.columns and delivery_col in df.columns:
        df[pickup_col] = pd.to_datetime(df[pickup_col], errors="coerce")
        df[delivery_col] = pd.to_datetime(df[delivery_col], errors="coerce")
        df["delivery_time_hours"] = (
            df[delivery_col] - df[pickup_col]
        ).dt.total_seconds() / 3600.0
    else:
        print("Warning: pickup/delivery columns missing. delivery_time_hours set to NaN.")
        df["delivery_time_hours"] = np.nan

    # Duplicate shipment flag
    if tracking_col in df.columns:
        df["duplicate_flag"] = df.duplicated(subset=[tracking_col], keep=False).astype(int)
    else:
        df["duplicate_flag"] = 0

    # Same city flag
    if source_col in df.columns and dest_col in df.columns:
        df["same_city"] = (
            df[source_col].astype(str).str.lower()
            == df[dest_col].astype(str).str.lower()
        ).astype(int)
    else:
        df["same_city"] = 0

    # Extra useful numeric features
    extra_features = [
        "actual_distance_to_destination",
        "actual_time",
        "osrm_time",
        "osrm_distance",
        "factor",
        "segment_actual_time",
        "segment_osrm_time",
        "segment_osrm_distance",
        "segment_factor",
    ]
    for col in extra_features:
        if col not in df.columns:
            df[col] = np.nan

    # Final fraud-relevant features
    features = [
        "delivery_time_hours",
        "actual_time",
        "osrm_time",
        "actual_distance_to_destination",
        "osrm_distance",
        "factor",
        "segment_factor",
        "is_cutoff",
        "duplicate_flag",
        "same_city"
    ]

    # Prepare feature matrix
    X = df[features].fillna(0).astype(float)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---------- IsolationForest ----------
    print(f"Training IsolationForest (contamination={contamination})...")
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    iso.fit(X_scaled)
    iso_pred = iso.predict(X_scaled)
    iso_scores = iso.decision_function(X_scaled)

    df["isolation_anomaly"] = (iso_pred == -1).astype(int)
    df["isolation_score"] = iso_scores
    df["Anomaly"] = df["isolation_anomaly"].map({0: "Normal", 1: "Fraud"})

    # Save anomaly results
    result_cols = [tracking_col, "isolation_anomaly", "isolation_score", "Anomaly"] + features
    if tracking_col not in df.columns:
        df.insert(0, "tracking_id_auto", ["row_" + str(i) for i in range(len(df))])
        result_cols.insert(0, "tracking_id_auto")

    out_df = df[result_cols]
    out_df.to_csv(out, index=False)
    print(f"Saved results (anomalies) to: {out}")

    # Show top anomalies
    top = df.sort_values("isolation_score").head(20)
    display_cols = [tracking_col, "isolation_score", "isolation_anomaly", "Anomaly"] + features
    print("\nTop 10 anomaly candidates (lowest isolation_score):")
    print(top[display_cols].head(10).to_string(index=False))

    # ---------- Optional supervised learning ----------
    if simulate_labels:
        print("\nSIMULATE LABELS: creating synthetic label...")
        if df["delivery_time_hours"].notna().sum() > 0:
            thr = df["delivery_time_hours"].quantile(0.95)
            df["sim_label"] = (
                (df["delivery_time_hours"] > thr) | (df["isolation_anomaly"] == 1)
            ).astype(int)
        else:
            df["sim_label"] = df["isolation_anomaly"].astype(int)

        from sklearn.model_selection import train_test_split

        mask = ~df[features].isnull().any(axis=1)
        X_sup = df.loc[mask, features].fillna(0).values
        y_sup = df.loc[mask, "sim_label"].values
        Xtr, Xte, ytr, yte = train_test_split(
            X_sup,
            y_sup,
            test_size=0.2,
            random_state=42,
            stratify=y_sup if len(np.unique(y_sup)) > 1 else None,
        )

        print("Training RandomForest on simulated labels...")
        rf = RandomForestClassifier(n_estimators=200, random_state=42)
        rf.fit(Xtr, ytr)
        ypred = rf.predict(Xte)
        print("\nRandomForest classification report on simulated labels:")
        print(classification_report(yte, ypred))

        model_path = os.path.join(os.path.dirname(out) or ".", "rf_sim_model.joblib")
        joblib.dump(rf, model_path)
        print("Saved simulated-label RF model to:", model_path)

    # ---------- Plots ----------
    try:
        plt.figure(figsize=(8, 4))
        sns.histplot(df["delivery_time_hours"].dropna(), bins=60, kde=True)
        plt.title("Delivery time (hours) distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(out) or ".", "delivery_time_dist.png"))
        print("Saved delivery_time_dist.png")

        plt.figure(figsize=(8, 5))
        sns.scatterplot(
            x="delivery_time_hours",
            y="actual_distance_to_destination",
            hue="isolation_anomaly",
            data=df,
            palette={0: "blue", 1: "red"},
            alpha=0.7,
        )
        plt.title("Anomaly (red) vs Normal (blue)")
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(out) or ".", "anomaly_scatter.png"))
        print("Saved anomaly_scatter.png")
    except Exception as e:
        print("Plotting skipped (error):", e)

    print("\nDone.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to Delhivery CSV")
    p.add_argument("--output", default="fraud_results.csv", help="Path to save results CSV")
    p.add_argument("--contamination", type=float, default=0.05, help="IsolationForest contamination (expected fraction of anomalies)")
    p.add_argument("--simulate_labels", action="store_true", help="Create simulated labels and train RF (for demo/evaluation)")
    args = p.parse_args()
    main(args)
