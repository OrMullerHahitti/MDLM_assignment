"""
Encoding utilities for converting preprocessed DataFrames to model-ready formats.

Different ML models require different feature encodings:
- Neural networks (MLP): One-hot encoding for categoricals, fill NaN with 0
- XGBoost: Label encoding with pandas Categorical, keep NaN
- CatBoost: String categoricals, keep NaN (native handling)
"""

import pandas as pd
from typing import Optional, Any


def encode_features_for_ml(
    df: pd.DataFrame,
    train_params: Optional[Any] = None
) -> pd.DataFrame:
    """
    Convert preprocessed DataFrame to ML-ready format for sklearn models.
    Uses one-hot encoding for nominal categoricals (needed for MLP/logistic regression).

    Args:
        df: Preprocessed DataFrame from preprocessing pipeline
        train_params: Optional FittedParams object with fitted encoders (deck_encoder, side_encoder)
                     If None, uses pd.factorize for ordinal encoding

    Returns:
        Encoded DataFrame ready for sklearn models
    """
    df = df.copy()

    # convert booleans to integers (handle both bool dtype and object with True/False)
    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
    for c in bool_cols:
        df[c] = df[c].astype(int)

    # also handle object columns that contain boolean-like values
    for c in ['CryoSleep', 'VIP']:
        if c in df.columns and df[c].dtype == 'object':
            df[c] = df[c].map({True: 1, False: 0, 'True': 1, 'False': 0}).fillna(0).astype(int)

    # label encode ordinal features (Deck, Side have natural ordering by position)
    if 'Deck' in df.columns:
        if train_params is not None and hasattr(train_params, 'deck_encoder'):
            try:
                df['Deck'] = train_params.deck_encoder.transform(df['Deck'])
            except Exception:
                df['Deck'] = pd.factorize(df['Deck'])[0]
        else:
            df['Deck'] = pd.factorize(df['Deck'])[0]

    if 'Side' in df.columns:
        if train_params is not None and hasattr(train_params, 'side_encoder'):
            try:
                df['Side'] = train_params.side_encoder.transform(df['Side'])
            except Exception:
                df['Side'] = pd.factorize(df['Side'])[0]
        else:
            df['Side'] = pd.factorize(df['Side'])[0]

    # encode Age_Group (already ordinal from binning)
    if 'Age_Group' in df.columns:
        df['Age_Group'] = pd.Categorical(df['Age_Group']).codes

    # one-hot encode nominal categorical features (for MLP/linear models)
    cat_cols = [c for c in ['HomePlanet', 'Destination', 'TravelAcompanyStatus'] if c in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # drop identifier columns
    for col in ('PassengerId', 'Name'):
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    # fill any remaining NaN with 0
    df = df.fillna(0)

    return df


def encode_features_for_xgboost(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert preprocessed DataFrame to XGBoost-optimized format.

    Key differences from general encoding:
    1. Label encode ALL categoricals (no one-hot) - trees split better this way
    2. Keep NaN values - XGBoost learns optimal direction for missing values
    3. Use pandas Categorical for native XGBoost categorical support

    Args:
        df: Preprocessed DataFrame from preprocessing pipeline

    Returns:
        Encoded DataFrame ready for XGBoost (with enable_categorical=True)
    """
    df = df.copy()

    # convert boolean columns to int (handle both bool dtype and object with True/False)
    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
    for c in bool_cols:
        df[c] = df[c].astype(int)

    # explicitly handle CryoSleep and VIP which may be object type with True/False values
    for c in ['CryoSleep', 'VIP', 'Transported']:
        if c in df.columns:
            if df[c].dtype == 'object' or df[c].dtype == 'bool':
                df[c] = df[c].map({True: 1, False: 0, 'True': 1, 'False': 0})
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)

    # convert categorical columns to pandas Categorical type
    cat_cols = ['Deck', 'Side', 'HomePlanet', 'Destination', 'Age_Group']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # drop identifier columns
    for col in ('PassengerId', 'Name', 'TravelAcompanyStatus'):
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    # IMPORTANT: do NOT fill NaN for numeric columns - XGBoost handles them natively!

    return df


def encode_features_for_catboost(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert preprocessed DataFrame to CatBoost-optimized format.

    CatBoost handles categoricals natively - we just need to:
    1. Convert boolean columns to int (CatBoost doesn't accept bool)
    2. Keep categorical columns as strings (CatBoost prefers this)
    3. Keep NaN values - CatBoost handles them natively

    Args:
        df: Preprocessed DataFrame from preprocessing pipeline

    Returns:
        Encoded DataFrame ready for CatBoost
    """
    df = df.copy()

    # convert boolean columns to int
    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
    for c in bool_cols:
        df[c] = df[c].astype(int)

    # handle object columns with True/False values
    for c in ['CryoSleep', 'VIP', 'Transported']:
        if c in df.columns:
            if df[c].dtype == 'object' or df[c].dtype == 'bool':
                df[c] = df[c].map({True: 1, False: 0, 'True': 1, 'False': 0})
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)

    # keep categorical columns as strings for CatBoost (it handles encoding internally)
    cat_cols = ['Deck', 'Side', 'HomePlanet', 'Destination', 'Age_Group']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # drop identifier columns
    for col in ('PassengerId', 'Name', 'TravelAcompanyStatus'):
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    return df


def get_catboost_categorical_indices(df: pd.DataFrame) -> list[int]:
    """
    Get indices of categorical columns for CatBoost's cat_features parameter.

    Args:
        df: DataFrame (without target column) to get categorical indices from

    Returns:
        List of column indices that are categorical
    """
    cat_cols = ['Deck', 'Side', 'HomePlanet', 'Destination', 'Age_Group']
    return [i for i, col in enumerate(df.columns) if col in cat_cols]
