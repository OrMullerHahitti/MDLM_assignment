"""
Feature engineering utility functions for the Spaceship Titanic dataset.
Extracted from Spaceship_Titanic.ipynb for reusability.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier


# columns representing passenger spending
SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def apply_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decomposes PassengerId and Cabin into their component features
    and removes the original high-cardinality columns.

    Args:
        df: the working DataFrame.

    Returns:
        the DataFrame after decomposition with new columns:
        - Group (from PassengerId)
        - Deck, CabinNum, Side (from Cabin)
    """
    df = df.copy()

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


def apply_feature_construction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs GroupSize, TotalSpent, NumSpendCategories, and TravelAcompanyStatus features.

    Args:
        df: the working DataFrame (must have 'Group' column from apply_decomposition).

    Returns:
        DataFrame with new constructed columns:
        - GroupSize: number of passengers in the same group
        - TotalSpent: sum of all spending columns
        - NumSpendCategories: count of spending categories with amount > 0
        - TravelAcompanyStatus: 'Solo', 'WithRelatives', or 'WithGroup'
    """
    df = df.copy()

    # calculate Group Size
    group_sizes = df.groupby('Group')['Group'].transform('count')
    df['GroupSize'] = group_sizes.astype('Int64', errors='ignore')

    # calculate Total Spent (NaNs in any spending column result in NaN)
    df['TotalSpent'] = df[SPEND_COLS].sum(axis=1, skipna=False)

    # calculate number of spending categories used (spent > 0)
    spend_has_nan = df[SPEND_COLS].isna().any(axis=1)
    df['NumSpendCategories'] = (df[SPEND_COLS] > 0).sum(axis=1).astype('Int64')
    df.loc[spend_has_nan, 'NumSpendCategories'] = pd.NA

    # calculate TravelAcompanyStatus
    # categories:
    # - Solo: GroupSize == 1
    # - WithRelatives: GroupSize > 1 and at least one other member has same surname
    # - WithGroup: GroupSize > 1 but no matching surname found
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

    # count how many passengers share this surname within the same group
    same_surname_counts = df.groupby(['Group', '_Surname'])['_Surname'].transform('count')
    has_relative = (df['GroupSize'] > 1) & (df['_Surname'].notna()) & (same_surname_counts >= 2)

    df['TravelAcompanyStatus'] = 'WithGroup'
    df.loc[df['GroupSize'] == 1, 'TravelAcompanyStatus'] = 'Solo'
    df.loc[has_relative, 'TravelAcompanyStatus'] = 'WithRelatives'

    # drop Name (and temp helper)
    df = df.drop(columns=['Name', '_Surname'], errors='ignore')

    return df


def apply_binning_and_visualize_max_leaf(
    df: pd.DataFrame,
    num_bins: int = 14,
    min_samples_per_bin: int = 150,
    min_purity_decrease: float = 0.00005,
    split_points: list | None = None,
    show: bool = True
) -> tuple[pd.DataFrame, list]:
    """
    Applies decision tree-based binning on Age column to create Age_Group.

    If split_points is provided, applies the same binning without re-fitting.
    Otherwise, fits a decision tree on Age vs Transported to learn optimal splits.

    Args:
        df: the working DataFrame (must have 'Age' column).
        num_bins: maximum number of bins (max_leaf_nodes for decision tree).
        min_samples_per_bin: minimum samples required per bin.
        min_purity_decrease: minimum impurity decrease for splitting.
        split_points: pre-computed split points (for test set). If None, learns from data.
        show: whether to display the visualization (only when learning split_points).

    Returns:
        tuple of (DataFrame with 'Age_Group' column, list of split_points).
    """
    df = df.copy()

    # if split points are not provided -> learn them from data (requires Transported)
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
        split_points = sorted(thresholds[thresholds > 0.0])

        print(f"Requested Max Bins (max_leaf_nodes): {num_bins}")
        print(f"Minimum Samples per Bin Required: {min_samples_per_bin}")
        print(f"Actual Optimal Split Points found: {split_points}")

    # define bins and labels based on the split points
    bins = [-np.inf] + list(split_points) + [np.inf]
    bin_labels = [f"Age_Bin_{i}" for i in range(len(split_points) + 1)]

    # apply the binning
    df['Age_Group'] = pd.cut(
        df['Age'],
        bins=bins,
        labels=bin_labels,
        right=True,
        include_lowest=True
    ).astype('object')

    # visualization only when we learned split_points (i.e., training data)
    if show and 'Transported' in df.columns and split_points is not None:
        age_stats = df.groupby('Age')['Transported'].agg(['mean', 'count']).reset_index()
        age_stats.columns = ['Age', 'Transported_Probability', 'N_per_Age']
        plot_df = age_stats.copy()

        bin_stats = df.groupby('Age_Group')['Transported'].agg(['mean', 'count']).sort_index()
        bin_n_counts = bin_stats['count']
        bin_mean_prob = bin_stats['mean']

        plt.figure(figsize=(14, 7))
        sns.lineplot(x='Age', y='Transported_Probability', data=plot_df,
                     marker='o', errorbar=None, zorder=2, color='darkgreen')

        plt.title(f"Mean Transported Probability by Age (Max Bins: {num_bins})", fontsize=16)
        plt.xlabel("Age", fontsize=14)
        plt.ylabel("Mean Transported Probability", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)

        # add Vertical Split Lines (Dashed)
        for split in split_points:
            plt.axvline(x=split, color='gray', linestyle='--', linewidth=1.5, zorder=1)
            plt.text(split, 0.95, f'Split: {split:.1f}',
                     rotation=90, va='center', ha='right', fontsize=9, color='darkred')

        # add Sample Size (N) Annotation per Bin
        bin_boundaries = [-0.5] + list(split_points) + [df['Age'].max() + 0.5]
        Y_OFFSET = 0.03

        for i, (label, count) in enumerate(bin_n_counts.items()):
            mid_point_x = (bin_boundaries[i] + bin_boundaries[i+1]) / 2
            dynamic_y = bin_mean_prob.loc[label] + Y_OFFSET
            if mid_point_x < 0:
                mid_point_x = (0 + bin_boundaries[i+1]) / 2

            plt.text(mid_point_x, dynamic_y,
                     f"N={count}",
                     ha='center', va='center',
                     fontsize=10,
                     color='blue',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))

        plt.xlim(0, plot_df['Age'].max() + 1)
        plt.xticks(np.arange(0, plot_df['Age'].max() + 5, 5))
        plt.tight_layout()
        plt.show()

    return df, split_points
