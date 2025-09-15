from flask import Flask, request, jsonify, send_file, send_from_directory
from backend import detect_fraud
import os
import pandas as pd
from werkzeug.utils import secure_filename
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return {"message": "Fraud Shipment Detection API running"}

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename)
    filepath = os.path.join("uploads", filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(filepath)

    # Run fraud detection
    output_csv = detect_fraud(filepath)

    # Load and return results as JSON
    df = pd.read_csv(output_csv)
    return df.to_json(orient="records")

@app.route('/results/<path:filename>')
def results_file(filename):
    # 'results' folder should be at project root or full path passed here
    return send_from_directory('results', filename)


@app.route("/results", methods=["GET"])
def get_results():
    output_csv = "results/fraud_results.csv"
    if not os.path.exists(output_csv):
        return jsonify({"error": "No results found"}), 404

    df = pd.read_csv(output_csv)
    return df.to_json(orient="records")

if __name__ == "__main__":
    app.run(debug=True)
