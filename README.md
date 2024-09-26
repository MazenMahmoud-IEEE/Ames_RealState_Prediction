# Ames Housing Price Prediction

This repository implements a machine learning project to predict housing prices using the Ames Housing dataset. The project involves data cleaning, preprocessing, model selection, and hyperparameter tuning to achieve optimal performance in predicting house sale prices.

## Project Overview

The goal of this project is to predict house sale prices based on various features in the Ames Housing dataset. The dataset consists of 2,931 records and 82 features, including information on the physical attributes of the houses, lot dimensions, utilities, and sale conditions.

### Dataset

- **Source**: Ames Housing Dataset
- **Size**: 2,931 rows, 82 columns
- **File Format**: TSV (Tab Separated Values)

## Project Workflow

### 1. Data Preprocessing and Cleaning

#### a. Importing Libraries
Key libraries used for data manipulation, visualization, and machine learning:

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
```

#### b. Data Loading and Copying
The dataset is loaded in TSV format and a copy is created for further manipulation:

```python
housing = pd.read_csv("Ames_Housing_Data1.tsv", sep='\t')
housing_1 = housing.copy()
```

#### c. Handling Missing Values
To prepare the data, missing values in key columns are addressed:

- Missing values in the `Lot Frontage` column are filled with the mean:

  ```python
  housing_1['Lot Frontage'].fillna(housing_1['Lot Frontage'].mean(), inplace=True)
  ```

- Missing values in columns related to garage attributes (`Garage Yr Blt`, `Garage Finish`, etc.) are handled appropriately.

#### d. Data Cleaning
The dataset is cleaned by removing duplicates, filling missing values, and handling outliers where necessary.

- For example, categorical variables are converted into numerical format for model compatibility, and numerical features are scaled using techniques like standardization.


### 2. Model Building and Tuning

After cleaning the data, multiple models are tested to predict house prices. The following models are explored:

#### a. Model Selection
- **Decision Tree**: A simple tree-based model used for regression.
- **Random Forest**: An ensemble method of decision trees used to improve performance.
- **Support Vector Machine (SVM)**: A robust regression method.
- **Gradient Boosting**: A powerful boosting algorithm for regression tasks.

After trying all models, **Random Forest** gave the best results after hyperparameter tuning. Other models were evaluated but did not perform as well.

#### b. Hyperparameter Tuning
Bayesian search and other techniques were used to fine-tune the Random Forest model to achieve optimal performance. The hyperparameters, such as the number of trees, maximum depth, and other settings, were adjusted.

### 3. Model Evaluation

The best-performing model, **Random Forest**, achieved an **R² score** of approximately **93%**, indicating a strong correlation between the predicted and actual house prices.



### 4. Results and Conclusion

After thorough evaluation, the Random Forest model stood out as the best option for predicting housing prices, achieving an impressive R² score of around 93%. Further steps could involve feature selection or additional tuning to push the performance even higher.

## Repository Structure

- `Ames_Housing_Prediction_Final.ipynb`: The main Jupyter notebook containing all code and steps from data preprocessing to model evaluation.
- `Ames_Housing_Data1.tsv`: The dataset (to be provided by the user).
- `README.md`: Project overview and instructions (this file).

## Installation and Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/Ames_Housing_Prediction.git
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebook to execute the project workflow:
   ```bash
   jupyter notebook Ames_Housing_Prediction_Final.ipynb
   ```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

