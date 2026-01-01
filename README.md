# Spaceship Titanic - Machine Learning Assignment

A comprehensive machine learning project for the [Kaggle Spaceship Titanic competition](https://www.kaggle.com/competitions/spaceship-titanic), predicting which passengers were transported to an alternate dimension during the Spaceship Titanic's collision with a spacetime anomaly.

## 📋 Project Overview

This project implements a complete machine learning pipeline including:
- Exploratory Data Analysis (EDA)
- Feature engineering and preprocessing
- Multiple classification models (Random Forest, XGBoost, CatBoost, MLP)
- Hyperparameter tuning with cross-validation
- Model evaluation and comparison

## 🗂️ Repository Structure

```
MDLM_assignment/
├── data/                          # Dataset files (train.csv, test.csv)
├── src/                           # Source code modules
│   ├── loading_and_preprocessing.py
│   ├── train.py
│   └── utils.py
├── utils/                         # Utility functions
│   ├── eda_utils.py              # Statistical analysis & visualization
│   ├── feature_utils.py          # Feature engineering functions
│   ├── preprocessing_pipeline.py # Data preprocessing pipeline
│   └── encoding_utils.py         # Feature encoding for ML models
├── Spaceship_Titanic_Understanding.ipynb  # EDA notebook
├── Spaceship_Titanic_Modeling.ipynb       # Model training notebook
├── pyproject.toml                # Project dependencies
├── uv.lock                       # Lock file for dependencies
└── README.md                     # This file
```

## 🔧 Installation

### Prerequisites
- Python >= 3.13

### Setup

1. Clone the repository:
```bash
git clone https://github.com/OrMullerHahitti/MDLM_assignment.git
cd MDLM_assignment
```

2. Install dependencies using `uv` or `pip`:
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### Dependencies
- **Data Processing**: `numpy`, `pandas`, `scipy`
- **Machine Learning**: `scikit-learn`, `xgboost`, `catboost`
- **Visualization**: `matplotlib`, `seaborn`
- **Development**: `ipykernel`, `black`, `pandas-stubs`, `scipy-stubs`
- **Data**: `kaggle` API client

## 📊 Data

The dataset comes from the [Spaceship Titanic Kaggle competition](https://www.kaggle.com/competitions/spaceship-titanic/data).

### Features
- **PassengerId**: Unique ID (format: gggg_pp where gggg is group)
- **HomePlanet**: Planet of departure (Earth, Europa, Mars)
- **CryoSleep**: Whether passenger was in suspended animation
- **Cabin**: Cabin number (format: deck/num/side)
- **Destination**: Planet of destination
- **Age**: Age of passenger
- **VIP**: Whether passenger paid for VIP service
- **RoomService, FoodCourt, ShoppingMall, Spa, VRDeck**: Amount billed at amenities
- **Name**: Passenger name
- **Transported**: Target variable (whether transported to alternate dimension)

## 🔬 Notebooks

### 1. Spaceship_Titanic_Understanding.ipynb
Comprehensive exploratory data analysis including:
- Statistical analysis (Spearman correlation, Cramér's V, Kruskal-Wallis tests)
- Missing value analysis
- Feature distributions and relationships
- Insights about spending patterns, demographics, and transportation outcomes

### 2. Spaceship_Titanic_Modeling.ipynb
Model training and evaluation:
- Data preprocessing pipeline
- Feature encoding strategies
- Multiple model implementations:
  - **Random Forest** (baseline ensemble method)
  - **XGBoost** (gradient boosting with label encoding)
  - **CatBoost** (handles categorical features natively)
  - **MLP** (neural network with custom architecture search)
- Hyperparameter tuning with RandomizedSearchCV
- Cross-validation and test set evaluation

## ⚙️ Preprocessing Pipeline

The preprocessing pipeline (`utils/preprocessing_pipeline.py`) includes:

1. **Feature Decomposition**
   - Split `PassengerId` into group IDs
   - Split `Cabin` into Deck, CabinNum, Side

2. **Feature Construction**
   - `GroupSize`: Number of passengers in the same travel group
   - `TotalSpent`: Sum of all amenity spending
   - `NumSpendCategories`: Count of amenities used
   - `TravelAcompanyStatus`: Solo, WithRelatives, or WithGroup

3. **Age Binning**
   - Convert continuous age into discrete bins for better model performance

4. **Imputation Strategy**
   - Spending and CryoSleep imputation (CryoSleep passengers have zero spending)
   - Categorical feature imputation using mode by group/deck
   - KNN imputation for remaining missing values

5. **Feature Recalculation**
   - Update derived features after imputation

## 🎯 Models and Performance

| Model | Best CV Accuracy | Test Accuracy | Key Parameters |
|-------|-----------------|---------------|----------------|
| Random Forest | 0.8036 | 0.8014 | n_estimators=436, max_depth=25 |
| XGBoost | 0.8110 | 0.8087 | max_depth=6, learning_rate=0.027 |
| CatBoost | 0.8136 | 0.8010 | depth=4, iterations=702 |
| MLP | ~0.80 | ~0.79 | Various architectures tested |

**Best performing model**: XGBoost with 80.87% test accuracy

## 🚀 Usage

### Running the Full Pipeline

```python
from utils.preprocessing_pipeline import preprocess_train_data, preprocess_test_data
import pandas as pd

# Load data
train_raw = pd.read_csv('data/train.csv')
test_raw = pd.read_csv('data/test.csv')

# Preprocess
train_processed, train_params = preprocess_train_data(train_raw)
test_processed = preprocess_test_data(test_raw, train_params)

# Train model (example with XGBoost)
from utils.encoding_utils import encode_features_for_xgboost
from xgboost import XGBClassifier

X = encode_features_for_xgboost(train_processed.drop('Transported', axis=1))
y = train_processed['Transported'].astype(int)

model = XGBClassifier(
    max_depth=6,
    learning_rate=0.027,
    enable_categorical=True
)
model.fit(X, y)
```

### Using Utility Functions

```python
# EDA utilities
from utils.eda_utils import run_spearman_analysis, run_cramers_v_analysis

# Feature engineering
from utils.feature_utils import apply_decomposition, apply_feature_construction

# Encoding for different models
from utils.encoding_utils import (
    encode_features_for_ml,      # One-hot encoding for most models
    encode_features_for_xgboost,  # Label encoding for XGBoost
    encode_features_for_catboost  # Special handling for CatBoost
)
```

## 📈 Key Insights

1. **CryoSleep is highly predictive**: Passengers in cryosleep have ~75% chance of being transported
2. **Spending patterns matter**: Zero spending correlates with higher transportation rates
3. **Home planet influences outcome**: Europa passengers more likely to be transported
4. **Group travel affects outcomes**: Solo travelers have different patterns than groups
5. **Deck location is important**: Certain decks (B, C) have higher transportation rates

## 🧪 Feature Engineering Highlights

- **Smart imputation**: CryoSleep logic ensures passengers in suspended animation have zero spending
- **Group features**: Leveraging travel group information improves predictions
- **Binned age**: Converting continuous age to discrete bins captures non-linear relationships
- **Cabin decomposition**: Splitting cabin into deck/number/side reveals spatial patterns

## 📝 Development Notes

- Code formatted with **Black**
- Type hints with **pandas-stubs** and **scipy-stubs**
- Modular design for easy experimentation
- Cross-validation ensures robust evaluation
- Hyperparameter tuning with RandomizedSearchCV for efficiency

## 🤝 Contributing

This is an assignment project, but suggestions and improvements are welcome!

## 📄 License

This project is open source and available for educational purposes.

## 🔗 Links

- [Kaggle Competition](https://www.kaggle.com/competitions/spaceship-titanic)
- [Repository](https://github.com/OrMullerHahitti/MDLM_assignment)

---

**Author**: OrMullerHahitti  
**Last Updated**: January 2026