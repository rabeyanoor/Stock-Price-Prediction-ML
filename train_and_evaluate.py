
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib

# Set random seed for reproducibility
np.random.seed(42)

def load_data(filepath):
    print("Loading data...")
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    print("\n--- Preprocessing Data ---")
    
    # Check for missing values
    print(f"Missing values before handling:\n{df.isnull().sum()}")
    
    # Drop rows where target (Stock_5) is missing, as we can't train on them
    if 'Stock_5' in df.columns:
        df = df.dropna(subset=['Stock_5'])
    
    # Handle missing values in Date if any (unlikely for index/date column usually)
    # Convert unnamed column to Data if needed, but likely the first column is Date
    # Looking at the file snippet, the first column seems to be index or empty, and second is Date?
    # Let's inspect the columns from the snippet:
    # 1: ,Stock_1,Stock_2,Stock_3,Stock_4,Stock_5
    # 2: 2020-01-01,101.7...
    # It seems the first column in the CSV is missing a header or is an empty string for the date column in the header row.
    # We will need to investigate the column names after loading.
    
    return df

def feature_engineering(df):
    print("\n--- Feature Engineering ---")
    # Identify the date column. Based on snippet, it might be the first column which might be unnamed.
    # We'll rename it to 'Date' if it's 'Unnamed: 0' or similar
    
    date_col = None
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            date_col = col
            break
    
    if date_col is None:
        # Heuristic: Check usually the first column if it looks like a date
        if df.iloc[0,0].startswith('20'):
           date_col = df.columns[0]
           print(f"Assuming first column '{date_col}' is the Date column.")
    
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df['Year'] = df[date_col].dt.year
        df['Month'] = df[date_col].dt.month
        df['Day'] = df[date_col].dt.day
        df['DayOfWeek'] = df[date_col].dt.dayofweek
        
        # Drop original Date column for modeling
        df = df.drop(columns=[date_col])
    else:
        print("Warning: Could not identify Date column. Skipping date-based features.")
        
    return df

def remove_outliers(df, columns):
    print("\n--- Outlier Detection ---")
    # Using IQR method
    initial_shape = df.shape
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    print(f"Removed outliers. Shape changed from {initial_shape} to {df.shape}")
    return df

def create_pipeline():
    print("\n--- Creating Pipeline ---")
    # Preprocessing for numerical data
    # We will handle imputation inside the pipeline just in case
    
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(random_state=42))
    ])
    
    return pipeline

def train_and_eval():
    file_path = 'stock_data.csv' 
    df = load_data(file_path)
    if df is None:
        return

    # Fix column names if needed (Handle the empty header for the date column)
    # Based on the snippet, the first column name is empty string or space in the file view?
    # View file output: "1: ,Stock_1,Stock_2,Stock_3,Stock_4,Stock_5"
    # This implies the first column name is likely empty or loaded as 'Unnamed: 0'
    
    if df.columns[0].strip() == '':
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
    elif 'Unnamed' in df.columns[0]:
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

    df = preprocess_data(df)
    df = feature_engineering(df)
    
    # Target and Features
    target_col = 'Stock_5'
    if target_col not in df.columns:
        print(f"Error: Target column {target_col} not found.")
        return

    # Features: All columns except target
    feature_cols = [c for c in df.columns if c != target_col]
    
    # Outlier detection on numerical features (stocks)
    stock_cols = [c for c in feature_cols if 'Stock' in c]
    df = remove_outliers(df, stock_cols + [target_col])
    
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"\nFeatures: {feature_cols}")
    print(f"Target: {target_col}")
    
    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Model Selection & Training
    pipeline = create_pipeline()
    print("Training Random Forest Regressor...")
    pipeline.fit(X_train, y_train)
    
    # Cross Validation
    print("\n--- Cross Validation ---")
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
    print(f"Cross-Validation R2 Scores: {cv_scores}")
    print(f"Average R2 Score: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    
    # Hyperparameter Tuning
    print("\n--- Hyperparameter Tuning ---")
    param_dist = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10]
    }
    
    random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=10, cv=3, scoring='r2', random_state=42, n_jobs=-1)
    random_search.fit(X_train, y_train)
    
    print(f"Best Parameters: {random_search.best_params_}")
    best_model = random_search.best_estimator_
    
    # Evaluation
    print("\n--- Specific Model Selection and Justification ---")
    print("Selected Model: Random Forest Regressor")
    print("Reason: Random Forests are robust to outliers, handle non-linear relationships well, and don't require heavy feature scaling (though we applied it). It is less prone to overfitting than a single Decision Tree.")

    print("\n--- Data Loading (5 Marks) ---")
    print(df.head())
    print(df.shape)
    
    print("\n--- Model Performance Evaluation ---")
    y_pred = best_model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    # Save the model features and pipeline
    print("\nSaving model to 'model.pkl'...")
    joblib.dump(best_model, 'model.pkl')
    # Also save the feature names validation
    joblib.dump(feature_cols, 'feature_cols.pkl')
    print("Model saved.")

if __name__ == "__main__":
    train_and_eval()
