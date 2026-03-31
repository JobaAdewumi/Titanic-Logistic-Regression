# Titanic Logistic Regression

This project uses logistic regression to predict passenger survival on the Titanic dataset. The full workflow lives in [Titanic logistic regression analysis.ipynb](https://github.com/JobaAdewumi/Titanic-Logistic-Regression/blob/main/Titanic%20logistic%20regression%20analysis.ipynb), from exploratory analysis through preprocessing, training, and evaluation.

## Project Files

- [Titanic logistic regression analysis.ipynb](https://github.com/JobaAdewumi/Titanic-Logistic-Regression/blob/main/Titanic%20logistic%20regression%20analysis.ipynb): main notebook containing the analysis and model training steps
- [titanic3.csv](https://github.com/JobaAdewumi/Titanic-Logistic-Regression/blob/main/titanic3.csv): dataset used by the notebook

## Workflow

The notebook follows this sequence:

1. Load `titanic3.csv` with pandas.
2. Explore the data with summary checks and plots for `survived`, `sex`, `pclass`, `age`, `fare`, and `sibsp`.
3. Inspect missing values.
4. Drop the high-missingness columns `body`, `cabin`, and `boat`.
5. Remove remaining rows with missing values using `dropna()`.
6. One-hot encode `sex`, `embarked`, and `pclass` with `drop_first=True`.
7. Drop non-numeric or unused columns: `name`, `sex`, `embarked`, `ticket`, `home.dest`, and the original `pclass`.
8. Split the data into training and test sets.
9. Train a `LogisticRegression` model with `solver='lbfgs'` and `max_iter=1000`.
10. Evaluate the model with a classification report, confusion matrix, and accuracy score.

## Dataset Notes

- Original dataset size: `1309` rows
- Columns in the raw dataset: `14`
- Missing values highlighted in the notebook:
  - `age`: `263`
  - `fare`: `1`
  - `embarked`: `2`
  - `cabin`: `1014`
  - `boat`: `823`
  - `body`: `1188`
  - `home.dest`: `564`
- Rows remaining after dropping `body`, `cabin`, `boat`, and then removing null rows: `684`

## Features Used

After preprocessing, the model is trained on these features:

- `age`
- `sibsp`
- `parch`
- `fare`
- `male`
- `Q`
- `S`
- `2`
- `3`

Target column:

- `survived`

## Results

The notebook records the following test-set metrics:

- Accuracy: `0.7767`
- Confusion matrix:

```text
[[91, 15],
 [31, 69]]
```

- Classification report:

```text
              precision    recall  f1-score   support

           0       0.75      0.86      0.80       106
           1       0.82      0.69      0.75       100

    accuracy                           0.78       206
   macro avg       0.78      0.77      0.77       206
weighted avg       0.78      0.78      0.77       206
```

## Requirements

To run the notebook locally, install a Python environment with:

- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`
- `scikit-learn`
- `jupyter`

Example:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn jupyter
```

Then start Jupyter and open the notebook:

```bash
jupyter notebook
```

## Notes

- This project is notebook-first; there is no standalone training script yet.
- The preprocessing strategy drops rows with missing values instead of imputing them.
- Feature names `2` and `3` come from one-hot encoding the passenger class column.
