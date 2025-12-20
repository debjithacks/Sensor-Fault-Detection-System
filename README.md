
# 🔧 Sensor Fault Detection System

An end-to-end **Machine Learning–based Sensor Fault Detection System** designed to automatically identify faulty and non-faulty sensor readings across multiple sensor types using a unified architecture.  
The system supports **Wafer, Gas, Temperature, Light (LDR), and Soil Moisture sensors** and provides real-time predictions through a Streamlit web interface.

🔗 **Live Demo:** https://sensor-fault-detection-system.streamlit.app/

---

## 📌 Project Highlights

- ✅ Multi-sensor fault detection using Machine Learning  
- ✅ All-in-One intelligent router for automatic sensor type detection  
- ✅ Robust preprocessing with imputation, scaling, and class balancing  
- ✅ Separate optimized models for each sensor type  
- ✅ Interactive Streamlit-based web interface  
- ✅ Production-ready modular project structure  

---

## 🧠 System Architecture

1. **Input Layer**  
   - CSV file upload or manual sensor input  

2. **All-in-One Router**  
   - Automatically detects sensor type based on schema  

3. **Preprocessing Pipeline**  
   - Missing value handling (KNN / Simple Imputer)  
   - Feature scaling (RobustScaler / StandardScaler)  
   - Class imbalance correction (SMOTETomek)  
   - Feature engineering  

4. **ML Model Layer**  
   - Wafer → Support Vector Classifier (SVC)  
   - Gas → Random Forest  
   - Temperature → Random Forest  
   - Light (LDR) → XGBoost  
   - Soil Moisture → XGBoost  

5. **Prediction & Visualization**  
   - Fault classification  
   - Downloadable results  
   - Visual insights  

---

## 📂 Supported Sensors

| Sensor Type | Fault Labels |
|------------|--------------|
| Wafer | 0 = Normal, 1 = Faulty |
| Gas | 0 = Normal, 1 = Faulty |
| Temperature | 0 = Faulty, 1 = Normal |
| Light (LDR) | -1 = Faulty, 1 = Normal |
| Soil Moisture | -1 = Faulty, 1 = Normal |

---

## 🛠️ Technologies Used

- **Programming Language:** Python  
- **Machine Learning:** Scikit-learn, XGBoost  
- **Data Processing:** NumPy, Pandas  
- **Imbalance Handling:** SMOTETomek  
- **Model Persistence:** Pickle (.pkl)  
- **Web Framework:** Streamlit  
- **Version Control:** Git & GitHub  

---

## 📁 Project Structure

```
Sensor-Fault-Detection/
│
├── data/                   # Sensor datasets
├── models/                 # Trained models (.pkl)
├── pipelines/              # Preprocessing pipelines
├── scripts/                # Training & utility scripts
├── custom_transformers.py  # Feature engineering logic
├── model_utils.py          # Model load/save utilities
├── all_in_one_router.py    # Sensor auto-detection logic
├── streamlit_app.py        # Web application
├── main.py                 # Training & prediction entry
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

---

## 🚀 How to Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/debjithacks/sensor-fault-detection.git
cd sensor-fault-detection
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application
```bash
streamlit run streamlit_app.py
```

---

## 📊 Model Evaluation

The models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Curve
- Confusion Matrix

All models achieved strong performance with ROC-AUC scores above **0.85**, confirming reliability for real-world usage.

---

## 👥 Team Members

This project was developed as a **team project** by:

- Aditya Maity  
- Souvik Pramanik  
- Avijit Ray  
- Subhadeep Hatai  
- **Debjit Ghosh**  

---

## 📌 Use Cases

- Industrial sensor health monitoring  
- Smart agriculture systems  
- IoT-based automation platforms  
- Preventive maintenance systems  

---

## 🔮 Future Enhancements

- Real-time sensor streaming (IoT integration)
- Deep learning-based fault detection
- Sensor fusion and anomaly explanation
- Cloud-native deployment

---

## 📜 License

This project is released under the **MIT License** – free to use, modify, and distribute with attribution.

---

⭐ If you find this project useful, please consider giving it a star!
