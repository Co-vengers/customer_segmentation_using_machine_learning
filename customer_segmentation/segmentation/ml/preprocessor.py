import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def preprocess_data(df, selected_columns=None):
    """
    Preprocess the data for clustering:
    1. Handle missing values
    2. Encode categorical variables
    3. Scale numerical features
    
    Args:
        df: pandas DataFrame with the original data
        selected_columns: List of columns to use (if None, use all columns)
        
    Returns:
        processed_df: Preprocessed DataFrame ready for clustering
        preprocessing_info: Dict with information about the preprocessing
    """
    # Use selected columns or all columns
    if selected_columns:
        df = df[selected_columns].copy()
    else:
        df = df.copy()
    
    # Identify numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    preprocessing_info = {
        'numerical_columns': numerical_cols,
        'categorical_columns': categorical_cols,
        'total_columns': len(numerical_cols) + len(categorical_cols),
        'rows_before': df.shape[0],
        'missing_values_per_column': df.isnull().sum().to_dict(),
    }
    
    # Create preprocessing pipeline
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Only add categorical transformer if we have categorical columns
    transformers = [
        ('num', numerical_transformer, numerical_cols)
    ]
    
    if categorical_cols:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', categorical_transformer, categorical_cols))
    
    preprocessor = ColumnTransformer(transformers=transformers)
    
    # Apply preprocessing
    processed_array = preprocessor.fit_transform(df)
    
    # Get feature names after transformation
    feature_names = numerical_cols.copy()
    
    if categorical_cols:
        # Get the one-hot encoder from the pipeline
        onehot_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        
        # Get the categories
        categories = onehot_encoder.categories_
        
        # Generate feature names for categorical variables
        onehot_feature_names = []
        for i, category in enumerate(categories):
            for cat_value in category:
                onehot_feature_names.append(f"{categorical_cols[i]}_{cat_value}")
        
        feature_names.extend(onehot_feature_names)
    
    # Convert to DataFrame
    processed_df = pd.DataFrame(processed_array, columns=feature_names, index=df.index)
    
    # Update preprocessing info
    preprocessing_info['rows_after'] = processed_df.shape[0]
    preprocessing_info['columns_after'] = processed_df.shape[1]
    
    return processed_df, preprocessing_info