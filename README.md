# 🚚 Fraud Shipment Detection System

A machine learning powered Flask web app to detect fraudulent shipments from CSV data.  
Frontend is built with HTML/CSS/JS, and backend with Flask + Python.  

---

## 📂 Project Structure

```
fraud-shipment-npn/
│── backend/
│   ├── app.py              # Main Flask application
│   ├── backend.py          # Helper functions
│   ├── fraud_detection.py  # ML fraud detection logic
│   ├── uploads/            # Uploaded CSV files
│   └── results/            # Processed output files
│
│── frontend/
│   ├── index.html          # Frontend dashboard
│   └── styles.css          # Dashboard styling
│
│── requirements.txt        # Python dependencies
│── venv/                   # Virtual environment
```

---

## ⚙️ How to Run the Project Locally

### 1. Create and activate a virtual environment
```bash
python -m venv venv
```

- On **Windows (PowerShell)**:
  ```bash
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  venv\Scripts\activate
  ```

- On **Mac/Linux**:
  ```bash
  source venv/bin/activate
  ```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the backend (Flask API)
Move into the `backend` folder and start Flask:
```bash
cd backend
flask run
```

The backend will run at:
```
http://127.0.0.1:5000/
```

### 4. Open the frontend
- Go to the `frontend` folder  
- Open `index.html` in your browser  

The frontend will connect to the backend and allow you to upload CSV files for fraud detection.

---

## 🧪 Usage

1. Prepare a CSV file with shipment data.  
   Example:
   ```csv
   trip_uuid,tracking_id_auto,shipment_id,delivery_time
   T12345,1001,S001,3.2
   T67890,1002,S002,15.6
   T54321,1003,S003,2.8
   ```

2. Open the dashboard (`index.html`).  
3. Upload the CSV file.  
4. Fraud detection results will appear in the interactive table.

---
