"""
House Price Prediction - Data Cleaning, EDA, Model Training & Export Script
Author: Senior AI/ML Engineer
Dataset: Kaggle 'House Price by Juhi Bhojani' (house_prices.csv, ~187k rows)
"""

import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

import sys

# Ensure UTF-8 output encoding for Windows terminal compatibility
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "house_prices.csv")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
BACKEND_MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend", "app", "models"))
FRONTEND_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "src", "data"))

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(BACKEND_MODELS_DIR, exist_ok=True)
os.makedirs(FRONTEND_DATA_DIR, exist_ok=True)


# ==========================================
# 1. PARSING & CLEANING HELPER FUNCTIONS
# ==========================================

def parse_amount(val):
    """
    Parses raw text amounts (e.g., '42 Lac', '1.2 Cr', 'Call for Price') into clean numeric rupees.
    Returns np.nan if unparseable or marked as 'Call for Price'.
    """
    if not isinstance(val, str) or val.strip() == '' or 'call for price' in val.lower():
        return np.nan
    v = val.lower().strip()
    try:
        if 'lac' in v:
            num = re.findall(r'[\d\.]+', v.replace('lac', ''))
            return float(num[0]) * 1e5 if num else np.nan
        elif 'cr' in v:
            num = re.findall(r'[\d\.]+', v.replace('cr', ''))
            return float(num[0]) * 1e7 if num else np.nan
        else:
            cleaned = re.sub(r'[^\d\.]', '', v)
            return float(cleaned) if cleaned else np.nan
    except Exception:
        return np.nan


def parse_area(val):
    """
    Extracts numeric values from area strings and converts to square feet (sqft).
    Normalizes 'sqm' (1 sqm ≈ 10.764 sqft) and 'sqyrd' (1 sqyrd = 9 sqft).
    """
    if not isinstance(val, str) or val.strip() == '':
        return np.nan
    v = val.lower().strip()
    try:
        nums = re.findall(r'[\d\.]+', v)
        if not nums:
            return np.nan
        num = float(nums[0])
        if 'sqm' in v or 'sq m' in v or 'sq.m' in v:
            num *= 10.764
        elif 'sqyrd' in v or 'sq yd' in v or 'sq yard' in v:
            num *= 9.0
        return num
    except Exception:
        return np.nan


def parse_floor(val):
    """
    Parses floor strings like '3 out of 10', 'Ground', 'Basement' into numeric integers.
    Ground -> 0, Basement -> -1.
    """
    if not isinstance(val, str) or val.strip() == '':
        return np.nan
    v = val.lower().strip()
    if 'basement' in v:
        return -1
    if 'ground' in v:
        return 0
    nums = re.findall(r'\d+', v)
    if nums:
        return int(nums[0])
    return np.nan


def parse_count(val):
    """Parses count fields like Bathroom/Balcony/Car Parking strings into numeric integers."""
    if not isinstance(val, str) or val.strip() == '':
        return np.nan
    v = val.lower().strip()
    nums = re.findall(r'\d+', v)
    if nums:
        return int(nums[0])
    return np.nan


def main():
    print("🚀 [1/6] Loading raw dataset...")
    df_raw = pd.read_csv(DATA_PATH)
    print(f"   Raw shape: {df_raw.shape[0]:,} rows, {df_raw.shape[1]} columns")

    print("🧹 [2/6] Parsing numeric features & cleaning target...")
    df = df_raw.copy()

    # Parse Price & Area
    df['price_clean'] = df['Amount(in rupees)'].apply(parse_amount).fillna(df['Price (in rupees)'])
    df['carpet_sqft'] = df['Carpet Area'].apply(parse_area)
    df['super_sqft'] = df['Super Area'].apply(parse_area)
    df['carpet_area_sqft'] = df['carpet_sqft'].fillna(df['super_sqft'])

    # Parse Floor, Bathroom, Balcony
    df['floor_num'] = df['Floor'].apply(parse_floor)
    df['bathroom'] = df['Bathroom'].apply(parse_count)
    df['balcony'] = df['Balcony'].apply(parse_count)

    # Drop missing target / area
    df_clean = df.dropna(subset=['price_clean', 'carpet_area_sqft']).copy()
    print(f"   Rows after dropping NaN target/area: {len(df_clean):,}")

    # Outlier Removal: Filter price_per_sqft outside 1st-99th percentile & area limits
    df_clean['price_per_sqft'] = df_clean['price_clean'] / df_clean['carpet_area_sqft']
    p1 = df_clean['price_per_sqft'].quantile(0.01)
    p99 = df_clean['price_per_sqft'].quantile(0.99)
    df_clean = df_clean[(df_clean['price_per_sqft'] >= p1) & (df_clean['price_per_sqft'] <= p99)]
    df_clean = df_clean[(df_clean['carpet_area_sqft'] >= 100) & (df_clean['carpet_area_sqft'] <= 15000)].copy()
    print(f"   Rows after 1st-99th percentile outlier filtering: {len(df_clean):,}")

    # High-Cardinality Location Handling (Keep top 50, rest -> 'other')
    top_locations = df_clean['location'].value_counts().head(50).index.tolist()
    df_clean['location_grouped'] = df_clean['location'].apply(lambda loc: loc if loc in top_locations else 'other')

    # Standardize string categories
    for col in ['Furnishing', 'Transaction', 'Ownership', 'facing']:
        df_clean[col] = df_clean[col].fillna('Unknown')

    print(f"   Top 50 locations identified. Total unique grouped locations: {df_clean['location_grouped'].nunique()}")

    # ==========================================
    # 2. EXPLORATORY DATA ANALYSIS (EDA) PLOTS
    # ==========================================
    print("📊 [3/6] Generating EDA Visualizations...")

    # Plot 1: Target Price Distribution (Log Scale)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df_clean['price_clean'], kde=True, log_scale=True, color='#3B82F6', ax=ax)
    ax.set_title("Distribution of Property Target Prices (Log Scale)", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Price in Rupees (Log Scale)", fontsize=11)
    ax.set_ylabel("Listing Count", fontsize=11)
    plt.tight_layout()
    plot1_path = os.path.join(PLOTS_DIR, "plot1_price_distribution.png")
    plt.savefig(plot1_path)
    plt.close()
    print(f"   Saved: {plot1_path}")

    # Plot 2: Price vs Carpet Area (Scatter Plot)
    fig, ax = plt.subplots(figsize=(8, 5))
    sample_df = df_clean.sample(n=min(5000, len(df_clean)), random_state=42)
    sns.regplot(data=sample_df, x='carpet_area_sqft', y='price_clean',
                scatter_kws={'alpha': 0.3, 'color': '#8B5CF6'}, line_kws={'color': '#EF4444', 'linewidth': 2}, ax=ax)
    ax.set_title("Price vs. Carpet Area (sqft)", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Carpet Area (sqft)", fontsize=11)
    ax.set_ylabel("Price in Rupees", fontsize=11)
    ax.set_yscale('log')
    plt.tight_layout()
    plot2_path = os.path.join(PLOTS_DIR, "plot2_price_vs_area.png")
    plt.savefig(plot2_path)
    plt.close()
    print(f"   Saved: {plot2_path}")

    # Plot 3: Top 15 Locations by Average Price
    fig, ax = plt.subplots(figsize=(10, 5))
    top15_loc = df_clean.groupby('location')['price_clean'].mean().sort_values(ascending=False).head(15) / 1e5
    sns.barplot(x=top15_loc.values, y=top15_loc.index, palette="viridis", ax=ax)
    ax.set_title("Top 15 Most Expensive Property Locations (Avg Price in Lacs)", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Average Price (in ₹ Lacs)", fontsize=11)
    ax.set_ylabel("Location", fontsize=11)
    plt.tight_layout()
    plot3_path = os.path.join(PLOTS_DIR, "plot3_top_locations.png")
    plt.savefig(plot3_path)
    plt.close()
    print(f"   Saved: {plot3_path}")

    # Plot 4: Price Distribution across Furnishing Status & Bathrooms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=df_clean, x='Furnishing', y='price_clean', palette="Set2", ax=ax1)
    ax1.set_yscale('log')
    ax1.set_title("Price Distribution by Furnishing Status", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Price in Rupees (Log Scale)")

    valid_bath = df_clean[df_clean['bathroom'].between(1, 6)]
    sns.boxplot(data=valid_bath, x='bathroom', y='price_clean', palette="Blues", ax=ax2)
    ax2.set_yscale('log')
    ax2.set_title("Price Distribution by Bathroom Count", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Price in Rupees (Log Scale)")

    plt.tight_layout()
    plot4_path = os.path.join(PLOTS_DIR, "plot4_furnishing_bathroom_boxplots.png")
    plt.savefig(plot4_path)
    plt.close()
    print(f"   Saved: {plot4_path}")

    # ==========================================
    # 3. PIPELINE SETUP & MODEL TRAINING
    # ==========================================
    print("🤖 [4/6] Building Scikit-Learn Pipeline & Training Models...")

    numeric_features = ['carpet_area_sqft', 'floor_num', 'bathroom', 'balcony']
    categorical_features = ['location_grouped', 'Furnishing', 'Transaction', 'Ownership', 'facing']

    X = df_clean[numeric_features + categorical_features]
    y = df_clean['price_clean']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_features)
    ])

    models = {
        "Linear Regression Baseline": TransformedTargetRegressor(
            regressor=Pipeline([
                ('prep', preprocessor),
                ('reg', LinearRegression())
            ]),
            func=np.log1p, inverse_func=np.expm1
        ),
        "HistGradientBoostingRegressor": TransformedTargetRegressor(
            regressor=Pipeline([
                ('prep', preprocessor),
                ('reg', HistGradientBoostingRegressor(max_iter=150, random_state=42))
            ]),
            func=np.log1p, inverse_func=np.expm1
        ),
        "RandomForestRegressor": TransformedTargetRegressor(
            regressor=Pipeline([
                ('prep', preprocessor),
                ('reg', RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1))
            ]),
            func=np.log1p, inverse_func=np.expm1
        )
    }

    results = {}
    best_model = None
    best_r2 = -float('inf')
    best_model_name = ""

    print("\n--- MODEL EVALUATION SUMMARY TABLE (TEST SET) ---")
    print(f"{'Model Name':<32} | {'MAE (₹)':<14} | {'RMSE (₹)':<14} | {'R² Score':<10}")
    print("-" * 78)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2, "model": model, "predictions": y_pred}
        print(f"{name:<32} | {mae:14,.2f} | {rmse:14,.2f} | {r2:10.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_model_name = name

    print("-" * 78)
    print(f"🏆 Winning Model: {best_model_name} with R² = {best_r2:.4f}\n")

    # Plot 5: Predicted vs Actual Scatter Plot for Best Model
    fig, ax = plt.subplots(figsize=(8, 6))
    y_test_sample = y_test.iloc[:3000]
    y_pred_sample = results[best_model_name]["predictions"][:3000]
    sns.scatterplot(x=y_test_sample, y=y_pred_sample, alpha=0.3, color='#10B981', ax=ax)
    max_val = max(y_test_sample.max(), y_pred_sample.max())
    ax.plot([0, max_val], [0, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(f"Predicted vs. Actual Prices ({best_model_name})", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Actual Price in Rupees (Log Scale)", fontsize=11)
    ax.set_ylabel("Predicted Price in Rupees (Log Scale)", fontsize=11)
    ax.legend()
    plt.tight_layout()
    plot5_path = os.path.join(PLOTS_DIR, "plot5_predicted_vs_actual.png")
    plt.savefig(plot5_path)
    plt.close()
    print(f"   Saved: {plot5_path}")

    # ==========================================
    # 4. CROSS VALIDATION EVALUATION (BONUS)
    # ==========================================
    print("🔁 [5/6] Performing 5-Fold Cross Validation on Winning Model...")
    cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='r2', n_jobs=-1)
    print(f"   5-Fold CV R² Scores: {[round(s, 4) for s in cv_scores]}")
    print(f"   Mean CV R²: {cv_scores.mean():.4f} (± {cv_scores.std():.4f})")

    # ==========================================
    # 5. EXPORT MODEL & LOCATIONS JSON
    # ==========================================
    print("💾 [6/6] Exporting Model Pipeline & Allowed Locations...")

    # Save model pipeline pickle
    model_export_path = os.path.join(MODELS_DIR, "house_price.pkl")
    backend_model_path = os.path.join(BACKEND_MODELS_DIR, "house_price.pkl")
    joblib.dump(best_model, model_export_path)
    joblib.dump(best_model, backend_model_path)
    print(f"   Saved fitted model pipeline to:")
    print(f"     - {model_export_path}")
    print(f"     - {backend_model_path}")

    # Sanity Check Reload
    reloaded_model = joblib.load(model_export_path)
    sample_input = X_test.iloc[[0]]
    sample_pred = reloaded_model.predict(sample_input)[0]
    print(f"   Sanity check prediction on sample: ₹ {sample_pred:,.2f}")

    # Save locations JSON list for frontend dropdowns
    allowed_locations = sorted(df_clean['location_grouped'].unique().tolist())
    locations_export_path = os.path.join(MODELS_DIR, "locations.json")
    backend_locations_path = os.path.join(BACKEND_MODELS_DIR, "locations.json")
    frontend_locations_path = os.path.join(FRONTEND_DATA_DIR, "locations.json")

    for loc_path in [locations_export_path, backend_locations_path, frontend_locations_path]:
        with open(loc_path, 'w', encoding='utf-8') as f:
            json.dump(allowed_locations, f, indent=2)
    print(f"   Saved {len(allowed_locations)} locations JSON to models/, backend/, and frontend/")

    print("\n✅ Data cleaning, EDA, training, evaluation, and export completed successfully!")


if __name__ == "__main__":
    main()
