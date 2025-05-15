# 🚗 Car Price Prediction Using XGBoost + Optuna + GPU

This project predicts car prices using machine learning. It uses **XGBoost** with **GPU acceleration** and **Optuna** for automated hyperparameter tuning. The model is trained on Persian car features, including brand, color, fuel type, gearbox, and more.

---

## 📁 Project Structure

```

├── train.py                          # Code to train and tune the model with Optuna
├── app.py                            # Use the saved model to predict from a local dictionary
├── data/car_price_model_xgb_gpu.pkl  # Saved model (after training)
├── README.md                         # Project documentation
├── data/car_price_model_xgb_gpu.pkl  # Saved model (after training)
├── data/car_data_cleaned.csv         # Normalized Data
└── data/car_data.csv                 # Example CSV (car listings with prices)

````

---

## 🧠 Features Used

- Year (`مدل (سال تولید)`)
- Insurance duration (`مهلت بیمهٔ شخص ثالث`)
- Mileage (`کارکرد`)
- One-hot encoded:
  - Brands (e.g., `برند_پژو`, `برند_207i`, `برند_TU5P`, etc.)
  - Colors (e.g., `رنگ_سفید`, `رنگ_مشکی`, `رنگ_آبی`, etc.)
  - Fuel Type, Gearbox Type

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install xgboost optuna scikit-learn pandas joblib
````

### 2. (Optional) Enable GPU

Ensure your environment supports CUDA and your version of XGBoost supports `gpu_hist`. If needed:

```bash
pip install xgboost --upgrade
```

### 3. Train the Model

```bash
python train.py
```

This will:

* Run hyperparameter tuning with Optuna
* Train the XGBoost model on the best parameters
* Save the model to disk as `car_price_model_xgb_gpu.pkl`

### 4. Predict on New Data

You can use `app.py` to predict the price of a single car defined in code:

```bash
python app.py
```

Make sure the features in the sample dictionary match the trained model.

---

## 🧪 Sample Prediction Output

```bash
Predicted car price: 763200000
```

---

## 📌 Notes

* All feature names are in Persian and match the CSV header.
* This project uses only numeric and one-hot encoded features.
* Preprocessing should match the format of the training data.

---

## 📃 License

This project is open-source and free to use under the MIT License.
