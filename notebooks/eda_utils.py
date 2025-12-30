"""
EDA utility functions for statistical analysis and visualization.
Extracted from Spaceship_Titanic.ipynb for reusability.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, kruskal


def run_spearman_analysis(df: pd.DataFrame, numeric_cols: list, binary_cols: list) -> pd.DataFrame:
    """
    Calculates and visualizes the Spearman correlation matrix for specified
    numerical and binary columns.

    Args:
        df: the input DataFrame.
        numeric_cols: list of numerical features.
        binary_cols: list of binary features (will be mapped to 0/1).

    Returns:
        the calculated Spearman correlation matrix.
    """
    # prepare data: select columns and map binary features to 0/1
    cols_to_analyze = numeric_cols + binary_cols
    spearman_df = df[cols_to_analyze].copy()

    for col in binary_cols:
        # convert boolean columns to integer type (True->1, False->0)
        if col in spearman_df.columns:
            spearman_df[col] = spearman_df[col].astype(int, errors='ignore')

    # calculate Spearman correlation matrix (based on ranked values)
    spearman_corr = spearman_df.corr(method="spearman")

    # visualization
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        spearman_corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0
    )
    plt.title("Spearman Correlation Matrix", fontsize=14)
    plt.tight_layout()
    plt.show()

    return spearman_corr


def cramers_v_statistic(confusion_matrix: np.ndarray) -> float:
    """
    Calculates Cramér's V statistic from a confusion matrix.
    Measures strength of association between two categorical variables.

    Args:
        confusion_matrix: contingency table as numpy array.

    Returns:
        Cramér's V value between 0 and 1.
    """
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    min_dim = min(confusion_matrix.shape[0] - 1, confusion_matrix.shape[1] - 1)
    if min_dim == 0 or n == 0:
        return 0.0
    return np.sqrt(chi2 / (n * min_dim))


def run_cramers_v_analysis(df: pd.DataFrame, categorical_features: list,show:bool = True) -> pd.DataFrame:
    """
    Computes pairwise Cramér's V for all categorical feature combinations
    and displays a heatmap.

    Args:
        df: the input DataFrame.
        categorical_features: list of categorical column names.

    Returns:
        DataFrame with Cramér's V values as a symmetric matrix.
    """
    n_features = len(categorical_features)
    cramers_matrix = np.zeros((n_features, n_features))

    for i, col1 in enumerate(categorical_features):
        for j, col2 in enumerate(categorical_features):
            if i == j:
                cramers_matrix[i, j] = 1.0
            elif i < j:
                # build contingency table
                contingency = pd.crosstab(df[col1], df[col2])
                v = cramers_v_statistic(contingency.values)
                cramers_matrix[i, j] = v
                cramers_matrix[j, i] = v

    cramers_df = pd.DataFrame(
        cramers_matrix,
        index=categorical_features,
        columns=categorical_features
    )

    if show:
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cramers_df,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="gray"
    )
        plt.title("Cramér's V: Categorical Feature Associations", fontsize=14)
        plt.tight_layout()
        plt.show()

    return cramers_df


def kruskal_test_with_effect_size(
    df: pd.DataFrame,
    cat_col: str,
    num_col: str
) -> tuple[float, float, float]:
    """
    Computes Kruskal-Wallis H-statistic, p-value, and normalized Effect Size (Epsilon Squared).
    This test determines if the median of num_col is significantly different across groups in cat_col.

    Args:
        df: the input DataFrame.
        cat_col: categorical column name (grouping variable).
        num_col: numerical column name (tested distribution).

    Returns:
        tuple of (H-statistic, p-value, effect_size). Returns (nan, nan, nan) if test cannot be run.
    """
    # subset data, dropping NaNs to ensure clean test
    subset = df[[cat_col, num_col]].dropna()
    N = len(subset)

    if N < 2:
        return np.nan, np.nan, np.nan

    # get the numerical data points for each group in the categorical column
    groups = [group[num_col].values for name, group in subset.groupby(cat_col)]
    k = len(groups)  # number of groups

    # must have at least two groups with data
    if k < 2 or any(len(g) == 0 for g in groups):
        return np.nan, np.nan, np.nan

    try:
        # run the Kruskal-Wallis H-test
        h_stat, p_value = kruskal(*groups)

        # calculate Effect Size (Epsilon Squared: eps2 = (H - (k - 1)) / (N - (k - 1)))
        numerator = h_stat - (k - 1)
        denominator = N - (k - 1)
        effect_size = numerator / denominator

        # effect size cannot be negative
        effect_size = max(0, effect_size)

        return h_stat, p_value, effect_size
    except ValueError:
        # fails if groups have identical values (rare)
        return np.nan, np.nan, np.nan


def run_kruskal_wallis_analysis(
    df: pd.DataFrame,
    cat_cols: list,
    num_cols: list
) -> pd.DataFrame:
    """
    Performs Kruskal-Wallis H-test for all categorical vs. numerical pairs
    and displays results as a sorted table and an Epsilon Squared (eps^2) heatmap.

    Args:
        df: the input DataFrame.
        cat_cols: list of categorical/binary features (grouping variables).
        num_cols: list of numerical features (tested distributions).

    Returns:
        DataFrame showing H-Statistic, P-value, and Effect Size (eps^2) for all pairs.
    """
    kruskal_results = []

    for cat_col in cat_cols:
        for num_col in num_cols:
            h_stat, p_value, effect_size = kruskal_test_with_effect_size(df, cat_col, num_col)

            kruskal_results.append({
                "Grouping_Feature": cat_col,
                "Tested_Numerical_Feature": num_col,
                "H_Statistic": h_stat,
                "Effect_Size": effect_size,
                "p_value": p_value
            })

    results_df = pd.DataFrame(kruskal_results).dropna(subset=['H_Statistic'])

    # sort and display the results table
    print("\n--- Kruskal-Wallis Results (Sorted by Effect Size) ---")
    display(results_df.sort_values("Effect_Size", ascending=False).reset_index(drop=True))

    # create the Effect Size Heatmap
    eta_matrix = results_df.pivot_table(
        index='Grouping_Feature',
        columns='Tested_Numerical_Feature',
        values='Effect_Size'
    ).loc[cat_cols, num_cols]  # ensure correct feature order

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        eta_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={'label': "Effect Size ($\\epsilon^2$)"},
    )

    plt.title("Kruskal-Wallis: Categorical vs. Numerical Effect Size ($\\epsilon^2$)", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    return results_df


# handle display() for non-jupyter environments
try:
    from IPython.display import display
except ImportError:
    def display(x):
        print(x)
