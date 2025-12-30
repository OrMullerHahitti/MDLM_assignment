"""
Preprocessing pipeline for Spaceship Titanic dataset.
Consolidates all transformations from the notebook into a single reusable pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from dataclasses import dataclass, field
from typing import Optional


SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


@dataclass
class FittedParams:
    """stores parameters learned from training data for consistent test transformation."""
    age_split_points: list = field(default_factory=list)
    deck_encoder: Optional[LabelEncoder] = None
    side_encoder: Optional[LabelEncoder] = None
    homeplanet_mode_by_deck: Optional[pd.Series] = None
    destination_mode_by_deck: Optional[pd.Series] = None
    age_group_mode_by_homeplanet: Optional[pd.Series] = None
    deck_mode_by_group: Optional[pd.Series] = None
    side_mode_by_group: Optional[pd.Series] = None
    cabinnum_mode_by_group: Optional[pd.Series] = None
    spending_medians: Optional[dict] = None


def _apply_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """decomposes PassengerId and Cabin into component features."""
    # decompose PassengerId into Group ID
    if 'PassengerId' in df.columns:
        df['Group'] = df['PassengerId'].str.split('_').str[0]
        df['Group'] = pd.to_numeric(df['Group'], errors='coerce').astype('Int64')

    # decompose Cabin into Deck, CabinNum, and Side
    if 'Cabin' in df.columns:
        df[['Deck', 'CabinNum', 'Side']] = df['Cabin'].str.split('/', expand=True)
        df['CabinNum'] = pd.to_numeric(df['CabinNum'], errors='coerce').astype('Int64')

    # drop original columns
    df = df.drop(columns=[col for col in ['PassengerId', 'Cabin'] if col in df.columns])

    return df


def _apply_feature_construction(df: pd.DataFrame) -> pd.DataFrame:
    """constructs GroupSize, TotalSpent, NumSpendCategories, TravelAcompanyStatus."""
    # calculate Group Size
    group_sizes = df.groupby('Group')['Group'].transform('count')
    df['GroupSize'] = group_sizes.astype('Int64', errors='ignore')

    # calculate Total Spent
    df['TotalSpent'] = df[SPEND_COLS].sum(axis=1, skipna=False)

    # calculate number of spending categories used
    spend_has_nan = df[SPEND_COLS].isna().any(axis=1)
    df['NumSpendCategories'] = (df[SPEND_COLS] > 0).sum(axis=1).astype('Int64')
    df.loc[spend_has_nan, 'NumSpendCategories'] = pd.NA

    # calculate TravelAcompanyStatus
    surnames = (
        df['Name']
        .fillna('')
        .astype(str)
        .str.strip()
        .str.split()
        .str[-1]
        .replace('', pd.NA)
    )
    df['_Surname'] = surnames

    same_surname_counts = df.groupby(['Group', '_Surname'])['_Surname'].transform('count')
    has_relative = (df['GroupSize'] > 1) & (df['_Surname'].notna()) & (same_surname_counts >= 2)

    df['TravelAcompanyStatus'] = 'WithGroup'
    df.loc[df['GroupSize'] == 1, 'TravelAcompanyStatus'] = 'Solo'
    df.loc[has_relative, 'TravelAcompanyStatus'] = 'WithRelatives'

    # drop Name and temp helper
    df = df.drop(columns=['Name', '_Surname'], errors='ignore')

    return df


def _apply_age_binning(
    df: pd.DataFrame,
    split_points: Optional[list] = None,
    num_bins: int = 14,
    min_samples_per_bin: int = 150,
    min_purity_decrease: float = 0.00005
) -> tuple[pd.DataFrame, list]:
    """applies decision tree-based age binning."""
    # learn split points from data if not provided
    if split_points is None:
        df_temp = df.dropna(subset=['Age', 'Transported']).copy()
        X = df_temp[['Age']]
        y = df_temp['Transported']

        tree = DecisionTreeClassifier(
            criterion='gini',
            random_state=42,
            max_leaf_nodes=num_bins,
            min_samples_leaf=min_samples_per_bin,
            min_impurity_decrease=min_purity_decrease
        )
        tree.fit(X, y)

        thresholds = tree.tree_.threshold[tree.tree_.feature == 0]
        split_points = sorted([float(t) for t in thresholds if t > 0.0])

    # apply binning
    bins = [-np.inf] + list(split_points) + [np.inf]
    bin_labels = [f"Age_Bin_{i}" for i in range(len(split_points) + 1)]

    df['Age_Group'] = pd.cut(
        df['Age'],
        bins=bins,
        labels=bin_labels,
        right=True,
        include_lowest=True
    ).astype('object')

    # drop Age column
    df = df.drop(columns=['Age'], errors='ignore')

    return df, split_points


def _impute_spending_and_cryosleep(df: pd.DataFrame) -> pd.DataFrame:
    """
    imputes spending and CryoSleep based on domain logic:
    - CryoSleep == True → spending should be 0
    - Positive spending → CryoSleep should be False
    - All zero spending → CryoSleep should be True
    """
    # step 1: impute missing spending as 0 for CryoSleep == True
    mask_cryo_true = df["CryoSleep"] == True
    df.loc[mask_cryo_true, SPEND_COLS] = df.loc[mask_cryo_true, SPEND_COLS].fillna(0)

    # step 2: impute CryoSleep = False when positive spending is observed
    mask_positive_spend = (df[SPEND_COLS] > 0).any(axis=1) & df["CryoSleep"].isna()
    df.loc[mask_positive_spend, "CryoSleep"] = False

    # step 3: impute remaining CryoSleep as True (all-zero spending cases)
    df.loc[df["CryoSleep"].isna(), "CryoSleep"] = True

    return df


def _compute_mode_maps(df: pd.DataFrame) -> dict:
    """computes mode mappings from training data for categorical imputation."""
    maps = {}

    # mode by group for cabin features
    for col in ['Deck', 'Side', 'CabinNum']:
        mode_map = (
            df.dropna(subset=[col, 'Group'])
            .groupby('Group')[col]
            .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None)
        )
        maps[f'{col}_by_group'] = mode_map

    # mode by deck for HomePlanet and Destination
    maps['HomePlanet_by_Deck'] = (
        df.dropna(subset=['HomePlanet', 'Deck'])
        .groupby('Deck')['HomePlanet']
        .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None)
    )

    maps['Destination_by_Deck'] = (
        df.dropna(subset=['Destination', 'Deck'])
        .groupby('Deck')['Destination']
        .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None)
    )

    # mode by HomePlanet for Age_Group
    maps['Age_Group_by_HomePlanet'] = (
        df.dropna(subset=['Age_Group', 'HomePlanet'])
        .groupby('HomePlanet')['Age_Group']
        .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None)
    )

    return maps


def _impute_by_mode(df: pd.DataFrame, target_col: str, group_col: str, mode_map: pd.Series) -> pd.DataFrame:
    """imputes target_col using mode_map based on group_col."""
    mask = df[target_col].isna() & df[group_col].notna()
    df.loc[mask, target_col] = df.loc[mask, group_col].map(mode_map)
    return df


def _knn_impute_categorical(
    df: pd.DataFrame,
    cols: list,
    encoders: Optional[dict] = None
) -> tuple[pd.DataFrame, dict]:
    """KNN impute for remaining missing values in categorical columns."""
    if encoders is None:
        encoders = {col: LabelEncoder().fit(df[col].dropna()) for col in cols}

    # encode categorical to numeric
    encoded = pd.DataFrame()
    for col in cols:
        encoded[col] = df[col].apply(
            lambda x: encoders[col].transform([x])[0] if pd.notna(x) else np.nan
        )

    # KNN impute
    imputer = KNNImputer(n_neighbors=5)
    imputed = pd.DataFrame(imputer.fit_transform(encoded), columns=cols, index=df.index)

    # decode back to categorical
    for col in cols:
        mask = df[col].isna()
        if mask.any():
            vals = np.clip(
                np.round(imputed.loc[mask, col]).astype(int),
                0,
                len(encoders[col].classes_) - 1
            )
            df.loc[mask, col] = encoders[col].inverse_transform(vals)

    return df, encoders


def _impute_categorical_features(
    df: pd.DataFrame,
    mode_maps: Optional[dict] = None,
    encoders: Optional[dict] = None,
    is_train: bool = True
) -> tuple[pd.DataFrame, dict, dict]:
    """
    imputes all categorical features using mode-based and KNN strategies.

    order:
    1. Deck, Side, CabinNum by Group mode, then KNN for remaining
    2. HomePlanet by Deck mode
    3. Destination by Deck mode
    4. Age_Group by HomePlanet mode
    5. VIP → False
    """
    # compute mode maps from training data
    if mode_maps is None and is_train:
        mode_maps = _compute_mode_maps(df)
    elif mode_maps is None:
        raise ValueError("mode_maps must be provided for test data")

    # step 0: cabin features by Group mode, then KNN
    for col in ['Deck', 'Side', 'CabinNum']:
        df = _impute_by_mode(df, col, 'Group', mode_maps.get(f'{col}_by_group', pd.Series()))

    # KNN for remaining Deck and Side
    if encoders is None:
        encoders = {}
    df, cat_encoders = _knn_impute_categorical(df, ['Deck', 'Side'], encoders.get('cat_encoders'))
    encoders['cat_encoders'] = cat_encoders

    # KNN for remaining CabinNum (numeric)
    if df['CabinNum'].isna().any():
        cabin_imputer = KNNImputer(n_neighbors=5)
        df['CabinNum'] = cabin_imputer.fit_transform(df[['CabinNum']])

    # step 1: HomePlanet by Deck mode
    df = _impute_by_mode(df, 'HomePlanet', 'Deck', mode_maps.get('HomePlanet_by_Deck', pd.Series()))

    # step 2: Destination by Deck mode
    df = _impute_by_mode(df, 'Destination', 'Deck', mode_maps.get('Destination_by_Deck', pd.Series()))

    # step 3: Age_Group by HomePlanet mode
    df = _impute_by_mode(df, 'Age_Group', 'HomePlanet', mode_maps.get('Age_Group_by_HomePlanet', pd.Series()))

    # step 4: VIP → False
    df['VIP'] = df['VIP'].fillna(False).infer_objects(copy=False)

    return df, mode_maps, encoders


def _impute_remaining_spending(df: pd.DataFrame, medians: Optional[dict] = None) -> tuple[pd.DataFrame, dict]:
    """imputes remaining spending NaNs with median values."""
    if medians is None:
        medians = {col: df[col].median() for col in SPEND_COLS}

    for col in SPEND_COLS:
        df[col] = df[col].fillna(medians[col])

    return df, medians


def _recalculate_spending_features(df: pd.DataFrame) -> pd.DataFrame:
    """recalculates TotalSpent and NumSpendCategories after spending imputation."""
    df['TotalSpent'] = df[SPEND_COLS].sum(axis=1, skipna=False)
    spend_has_nan = df[SPEND_COLS].isna().any(axis=1)
    df['NumSpendCategories'] = (df[SPEND_COLS] > 0).sum(axis=1).astype('Int64')
    df.loc[spend_has_nan, 'NumSpendCategories'] = pd.NA
    return df


def preprocess_train(
    df: pd.DataFrame,
    impute_spending: bool = True,
    verbose: bool = True
) -> tuple[pd.DataFrame, FittedParams]:
    """
    preprocesses training data and returns fitted parameters for test transformation.

    pipeline steps:
    1. decompose PassengerId and Cabin
    2. construct GroupSize, TotalSpent, NumSpendCategories, TravelAcompanyStatus
    3. bin Age into Age_Group (learns split points)
    4. impute spending and CryoSleep based on domain logic
    5. impute categorical features (learns mode maps)
    6. impute remaining spending with median (optional)
    7. recalculate spending features

    Args:
        df: training DataFrame (must have 'Transported' column)
        impute_spending: whether to impute remaining spending NaNs with median
        verbose: whether to print progress messages

    Returns:
        tuple of (processed DataFrame, FittedParams for test transformation)
    """
    df = df.copy()
    params = FittedParams()

    if verbose:
        print("Step 1: Decomposing PassengerId and Cabin...")
    df = _apply_decomposition(df)

    if verbose:
        print("Step 2: Constructing features (GroupSize, TotalSpent, etc.)...")
    df = _apply_feature_construction(df)

    if verbose:
        print("Step 3: Binning Age into Age_Group...")
    df, params.age_split_points = _apply_age_binning(df)

    if verbose:
        print("Step 4: Imputing spending and CryoSleep...")
    df = _impute_spending_and_cryosleep(df)

    if verbose:
        print("Step 5: Imputing categorical features...")
    df, mode_maps, encoders = _impute_categorical_features(df, is_train=True)

    # store fitted parameters
    params.homeplanet_mode_by_deck = mode_maps.get('HomePlanet_by_Deck')
    params.destination_mode_by_deck = mode_maps.get('Destination_by_Deck')
    params.age_group_mode_by_homeplanet = mode_maps.get('Age_Group_by_HomePlanet')
    params.deck_mode_by_group = mode_maps.get('Deck_by_group')
    params.side_mode_by_group = mode_maps.get('Side_by_group')
    params.cabinnum_mode_by_group = mode_maps.get('CabinNum_by_group')
    if 'cat_encoders' in encoders:
        params.deck_encoder = encoders['cat_encoders'].get('Deck')
        params.side_encoder = encoders['cat_encoders'].get('Side')

    if impute_spending:
        if verbose:
            print("Step 6: Imputing remaining spending with median...")
        df, params.spending_medians = _impute_remaining_spending(df)

    if verbose:
        print("Step 7: Recalculating spending features...")
    df = _recalculate_spending_features(df)

    if verbose:
        print("Done! Training data preprocessed.")

    return df, params


def preprocess_test(
    df: pd.DataFrame,
    params: FittedParams,
    verbose: bool = True
) -> pd.DataFrame:
    """
    preprocesses test data using parameters fitted on training data.

    Args:
        df: test DataFrame
        params: FittedParams from preprocess_train()
        verbose: whether to print progress messages

    Returns:
        processed DataFrame
    """
    df = df.copy()

    if verbose:
        print("Step 1: Decomposing PassengerId and Cabin...")
    df = _apply_decomposition(df)

    if verbose:
        print("Step 2: Constructing features (GroupSize, TotalSpent, etc.)...")
    df = _apply_feature_construction(df)

    if verbose:
        print("Step 3: Applying Age binning (using train split points)...")
    df, _ = _apply_age_binning(df, split_points=params.age_split_points)

    if verbose:
        print("Step 4: Imputing spending and CryoSleep...")
    df = _impute_spending_and_cryosleep(df)

    if verbose:
        print("Step 5: Imputing categorical features...")
    mode_maps = {
        'Deck_by_group': params.deck_mode_by_group,
        'Side_by_group': params.side_mode_by_group,
        'CabinNum_by_group': params.cabinnum_mode_by_group,
        'HomePlanet_by_Deck': params.homeplanet_mode_by_deck,
        'Destination_by_Deck': params.destination_mode_by_deck,
        'Age_Group_by_HomePlanet': params.age_group_mode_by_homeplanet,
    }
    encoders = {
        'cat_encoders': {
            'Deck': params.deck_encoder,
            'Side': params.side_encoder,
        }
    }
    df, _, _ = _impute_categorical_features(df, mode_maps=mode_maps, encoders=encoders, is_train=False)

    if params.spending_medians is not None:
        if verbose:
            print("Step 6: Imputing remaining spending with train medians...")
        df, _ = _impute_remaining_spending(df, medians=params.spending_medians)

    if verbose:
        print("Step 7: Recalculating spending features...")
    df = _recalculate_spending_features(df)

    if verbose:
        print("Done! Test data preprocessed.")

    return df


def preprocess_spaceship_data(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    impute_spending: bool = True,
    verbose: bool = True
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], FittedParams]:
    """
    convenience function to preprocess both train and test data.

    Args:
        train_df: training DataFrame (must have 'Transported' column)
        test_df: test DataFrame (optional)
        impute_spending: whether to impute remaining spending NaNs with median
        verbose: whether to print progress messages

    Returns:
        tuple of (processed train, processed test or None, FittedParams)

    Example:
        train_processed, test_processed, params = preprocess_spaceship_data(train, test)
    """
    if verbose:
        print("=" * 50)
        print("PREPROCESSING TRAINING DATA")
        print("=" * 50)

    train_processed, params = preprocess_train(train_df, impute_spending=impute_spending, verbose=verbose)

    test_processed = None
    if test_df is not None:
        if verbose:
            print("\n" + "=" * 50)
            print("PREPROCESSING TEST DATA")
            print("=" * 50)
        test_processed = preprocess_test(test_df, params, verbose=verbose)

    return train_processed, test_processed, params
