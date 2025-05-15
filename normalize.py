import pandas as pd
import re

# === Utility functions ===

def convert_to_english_digits(text):
    if pd.isna(text):
        return text
    persian_digits = '۰۱۲۳۴۵۶۷۸۹'
    arabic_digits = '٠١٢٣٤٥٦٧٨٩'
    english_digits = '0123456789'
    for p, e in zip(persian_digits, english_digits):
        text = text.replace(p, e)
    for a, e in zip(arabic_digits, english_digits):
        text = text.replace(a, e)
    return text

def clean_price(price_str):
    if pd.isna(price_str):
        return None
    price_str = convert_to_english_digits(price_str)
    price_str = price_str.replace("تومان", "").replace(",", "").replace("٬", "").strip()
    try:
        return int(price_str)
    except:
        return None

def clean_km(km_str):
    if pd.isna(km_str):
        return 0
    km_str = convert_to_english_digits(km_str)
    return int(km_str.replace(",", "").replace("٬", "").strip())

def extract_months(text):
    if pd.isna(text):
        return 0
    text = convert_to_english_digits(text)
    return int(text.replace("ماه", "").strip())

def tokenize_brand_type(text):
    if pd.isna(text):
        return []
    return text.split()

def has_207(text):
    return "207" in text or "۲۰۷" in text

def is_invalid_price(price):
    return bool(re.fullmatch(r"(.)\1{1,}", str(price)))

# === Load and preprocess ===

df = pd.read_csv("data/car_data.csv")
print(f"Initial rows: {len(df)}")

# Clean numeric/text columns
df['قیمت پایه'] = df['قیمت پایه'].apply(clean_price)
df['کارکرد'] = df['کارکرد'].apply(clean_km)
df['مدل (سال تولید)'] = df['مدل (سال تولید)'].apply(lambda x: int(convert_to_english_digits(str(x))) if pd.notna(x) else 0)
df['مهلت بیمهٔ شخص ثالث'] = df['مهلت بیمهٔ شخص ثالث'].apply(extract_months)

# Drop unused column
df.drop(columns=['مایل به معاوضه'], inplace=True)

# Filter rows step-by-step
df = df[df['برند و تیپ'].apply(has_207)]
print(f"After 207 filter: {len(df)}")

df = df[~df['قیمت پایه'].apply(is_invalid_price)]
print(f"After invalid price pattern filter: {len(df)}")

# ✅ Updated price range: 450,000,000 – 1,200,000,000
df = df[(df['قیمت پایه'] >= 450000000) & (df['قیمت پایه'] <= 1200000000)]
print(f"After updated price range filter: {len(df)}")

# Tokenize and one-hot encode brand tokens
df['توکن برند و تیپ'] = df['برند و تیپ'].apply(tokenize_brand_type)
token_set = set(token for tokens in df['توکن برند و تیپ'] for token in tokens)
for token in token_set:
    df[f'برند_{token}'] = df['توکن برند و تیپ'].apply(lambda tokens: int(token in tokens))
df.drop(columns=['توکن برند و تیپ'], inplace=True)

# ✅ Remove original برند و تیپ column
df.drop(columns=['برند و تیپ'], inplace=True)

# Now safely one-hot encode remaining rows for رنگ, نوع سوخت, گیربکس
categorical_columns = ['رنگ', 'نوع سوخت', 'گیربکس']
df[categorical_columns] = df[categorical_columns].fillna('نامشخص')
df_encoded = pd.get_dummies(df[categorical_columns], prefix=categorical_columns, dtype=int)
df.drop(columns=categorical_columns, inplace=True)

# Final concat
df_final = pd.concat([df, df_encoded], axis=1)

# Save output
df_final.to_csv("data/car_data_cleaned.csv", index=False)
print(f"✅ Final saved: {len(df_final)} rows, {df_final.shape[1]} columns")
