import pandas as pd
import joblib

# Load the saved model
model = joblib.load("car_price_model_xgb_gpu.pkl")

# Example single car data as a dict (use your actual feature names and values)
sample_data = {
    "مدل_(سال_تولید)": 1404,
    "مهلت_بیمه_شخص_ثالث": 0,
    "کارکرد": 0,
    "برند_پژو": 0,
    "برند_TU3": 0,
    "برند_دنده_ای": 0,
    "برند_SD": 0,
    "برند_پانوراما": 0,
    "برند_اتوماتیک": 1,
    "برند_TU5P": 0,
    "برند_MC": 0,
    "برند_207i": 0,
    "رنگ_آبی": 0,
    "رنگ_آلبالویی": 0,
    "رنگ_بادمجانی": 0,
    "رنگ_بنفش": 0,
    "رنگ_تیتانیوم": 0,
    "رنگ_خاکستری": 0,
    "رنگ_دلفینی": 0,
    "رنگ_ذغالی": 0,
    "رنگ_سبز": 0,
    "رنگ_سربی": 0,
    "رنگ_سرمه_ای": 0,
    "رنگ_سفید": 0,
    "رنگ_سفید_صدفی": 0,
    "رنگ_قرمز": 0,
    "رنگ_مسی": 0,
    "رنگ_مشکی": 0,
    "رنگ_نقره_ای": 0,
    "رنگ_نوک_مدادی": 0,
    "رنگ_کربن_بلک": 0,
    "رنگ_گیلاسی": 0,
    "نوع_سوخت_بنزینی": 0,
    "گیربکس_اتوماتیک": 1,
    "گیربکس_دنده_ای": 0,
    "گیربکس_نامشخص": 0
}

# Convert to DataFrame with one row (model expects 2D)
sample_df = pd.DataFrame([sample_data])

# Predict price
predicted_price = model.predict(sample_df)

print(f"Predicted car price: {int(predicted_price[0])}")
