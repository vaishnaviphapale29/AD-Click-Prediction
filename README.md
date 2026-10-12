# AD Click Prediction System

## Project Overview
The **AD Click Prediction System** is a Machine Learning classification project that predicts whether a user will click on an online advertisement based on user demographics, device information, advertisement placement, browsing history, and time of day.

The project compares multiple machine learning models and selects the best-performing model using ROC-AUC score.

## Objectives
- Predict whether a user will click on an advertisement.
- Handle missing values using a preprocessing pipeline.
- Perform Exploratory Data Analysis (EDA).
- Compare multiple classification models.
- Evaluate model performance using classification metrics.
- Save and reload the trained model for future predictions.

## Dataset Information
- **Total Records:** 10,000
- **Input Features:** 6
- **Training Records:** 8,000
- **Testing Records:** 2,000
- **Target Column:** `click`

### Input Features
- `age`
- `gender`
- `device_type`
- `ad_position`
- `browsing_history`
- `time_of_day`

### Target
- `1` – Advertisement clicked
- `0` – Advertisement not clicked

The dataset contains missing values in several input columns. These are handled by the machine learning preprocessing pipeline.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib

## Machine Learning Workflow
1. Load the dataset.
2. Inspect the dataset and check missing values.
3. Perform statistical analysis and EDA.
4. Visualize click distribution and click rates by browsing history.
5. Separate input features and target.
6. Split the dataset into training and testing sets.
7. Create the preprocessing pipeline.
8. Train and compare machine learning models.
9. Evaluate the models.
10. Select the best model using ROC-AUC.
11. Generate predictions.
12. Save and reload the trained model.

## Models Implemented
- Logistic Regression
- Logistic Regression (Balanced)
- Random Forest (Balanced)

## Model Comparison Results

| Model | Accuracy | Macro F1 | Test ROC-AUC | CV ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 65.00% | 0.3953 | 0.5554 | 0.5455 |
| Logistic Regression (Balanced) | 53.90% | 0.5251 | 0.5555 | 0.5457 |
| Random Forest (Balanced) | 67.85% | 0.6425 | 0.6994 | 0.6771 |

**Best Model:** Random Forest (Balanced)

The Random Forest (Balanced) model was selected based on the highest Test ROC-AUC score of **0.6994**.

## Final Model Evaluation

| Metric | Result |
|---|---:|
| Accuracy | 67.85% |
| Precision | 0.6749 |
| Recall | 0.6785 |
| F1 Score | 0.6765 |
| Macro F1 Score | 0.6425 |
| ROC-AUC Score | 0.6994 |

### Classification Report

| Class | Precision | Recall | F1 Score | Support |
|---|---:|---:|---:|---:|
| No Click (0) | 0.54 | 0.52 | 0.53 | 700 |
| Click (1) | 0.75 | 0.77 | 0.76 | 1300 |

### Confusion Matrix

```text
[[361 339]
 [304 996]]
```

- True Negatives: 361
- False Positives: 339
- False Negatives: 304
- True Positives: 996

## Generated Files

The project generates the following output files:

### Visualizations
- `class_distribution.png` – Displays the distribution of ad clicks.
- `ad_click_reasons.png` – Shows click rates across browsing-history categories.
- `confusion_matrix.png` – Visualizes correct and incorrect predictions.
- `roc_curve.png` – Shows the model's ROC curve.

### Model and Results
- `prediction_results.csv` – Contains prediction results.
- `ad_click_model.pkl` – Saved trained machine learning model.

## How to Run the Project

### 1. Install Dependencies

```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

### 2. Run the Python Script

```bash
python AD_Click_Prediction.py
```

Use the Python interpreter where the required dependencies are installed.

## Key Findings
- The dataset contains 10,000 records.
- The target distribution is 65% clicks and 35% non-clicks.
- Missing values are present in several input features.
- Random Forest (Balanced) achieved the highest Test ROC-AUC among the three evaluated models.
- The saved model was successfully loaded and tested.

## Conclusion
The AD Click Prediction System demonstrates an end-to-end machine learning workflow, including data preprocessing, exploratory data analysis, model comparison, evaluation, prediction, and model saving.

The best model achieved **67.85% accuracy** and a **0.6994 ROC-AUC score** on the reported test evaluation. Further tuning and feature engineering may improve predictive performance.

## Future Improvements
- Tune model hyperparameters.
- Improve prediction performance for the non-click class.
- Experiment with additional classification algorithms.
- Analyze feature importance.
- Test the model on new user data.