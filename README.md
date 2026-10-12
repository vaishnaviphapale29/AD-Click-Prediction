# AD Click Prediction

## Project Overview

The **AD Click Prediction** project is a Machine Learning classification project designed to predict whether a user will click on an advertisement based on demographic information and browsing behavior.

The project follows an end-to-end Machine Learning workflow, including data loading, data exploration, missing value handling, preprocessing, model training, model comparison, evaluation, visualization, and model persistence.

## Objectives

- Analyze advertisement click behavior using user-related features.
- Handle missing values through preprocessing pipelines.
- Train and compare multiple classification models.
- Evaluate model performance using classification metrics.
- Select the best-performing model based on Test ROC-AUC.
- Save and reload the trained model for future predictions.

## Dataset Information

- **Total Records:** 10,000
- **Original Columns:** 9
- **Input Features Used:** 6
- **Target Variable:** `click`
- **Training Records:** 8,000
- **Testing Records:** 2,000

### Input Features

| Feature | Description |
|---|---|
| `age` | User's age |
| `gender` | User's gender |
| `device_type` | Device used by the user |
| `ad_position` | Position of the advertisement |
| `browsing_history` | User's browsing category |
| `time_of_day` | Time when the advertisement is shown |

**Target variable:** `click`

- `0` — Advertisement not clicked
- `1` — Advertisement clicked

The `id` and `full_name` columns were excluded from the model inputs.

## Technologies Used

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## Machine Learning Workflow

1. Load the dataset.
2. Inspect the dataset and identify missing values.
3. Perform statistical analysis and examine the target distribution.
4. Visualize click distribution and click rates by browsing history.
5. Separate input features and target variable.
6. Split the dataset into training and testing sets using stratified sampling.
7. Build preprocessing pipelines for numerical and categorical features.
8. Train multiple classification models.
9. Compare models using accuracy, Macro F1, class 0 recall, Test ROC-AUC, and cross-validation ROC-AUC.
10. Select the best model and generate predictions.
11. Evaluate the selected model using classification metrics, a confusion matrix, and an ROC curve.
12. Save the trained model and verify it by loading it again.

## Data Preprocessing

The project uses a Scikit-learn preprocessing pipeline.

- **Numerical features:** Missing values are filled using the median, followed by standard scaling.
- **Categorical features:** Missing values are filled using the most frequent value, followed by one-hot encoding.
- **Unknown categories:** Handled using `OneHotEncoder(handle_unknown="ignore")`.

The preprocessing steps are combined with the classifiers in a pipeline.

## Models Trained

Three classification models were evaluated:

1. Logistic Regression
2. Logistic Regression with balanced class weights
3. Random Forest with balanced class weights

## Model Comparison Results

The following results were obtained from the project execution.

| Model | Accuracy | Macro F1 | Class 0 Recall | Test ROC-AUC | CV ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6500 | 0.3953 | 0.0014 | 0.5554 | 0.5455 |
| Logistic Regression (Balanced) | 0.5390 | 0.5251 | 0.5257 | 0.5555 | 0.5457 |
| Random Forest (Balanced) | **0.6785** | **0.6425** | 0.5157 | **0.6994** | **0.6771** |

**Selected Model: Random Forest (Balanced)**

The Random Forest model achieved the highest Test ROC-AUC among the three evaluated models.

## Final Model Evaluation

| Metric | Result |
|---|---:|
| Accuracy | 67.85% |
| Weighted Precision | 67.49% |
| Weighted Recall | 67.85% |
| Weighted F1 Score | 67.65% |
| Macro F1 Score | 64.25% |
| ROC-AUC Score | 69.94% |

### Classification Report

| Class | Precision | Recall | F1 Score | Support |
|---|---:|---:|---:|---:|
| 0 — Not Clicked | 0.54 | 0.52 | 0.53 | 700 |
| 1 — Clicked | 0.75 | 0.77 | 0.76 | 1,300 |
| Macro Average | 0.64 | 0.64 | 0.64 | 2,000 |
| Weighted Average | 0.67 | 0.68 | 0.68 | 2,000 |

### Confusion Matrix

The confusion matrix, with class order `[0, 1]`, is:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 361 | 339 |
| Actual 1 | 304 | 996 |

The model correctly classified 361 non-clicks and 996 clicks. It incorrectly predicted 339 non-clicks as clicks and missed 304 actual clicks.

### Click Rate by Browsing History

| Browsing History | Click Rate |
|---|---:|
| Entertainment | 68.77% |
| Shopping | 64.53% |
| Social Media | 63.57% |
| Education | 62.78% |
| News | 59.12% |

In this dataset, the Entertainment browsing category had the highest observed click rate.

## Generated Outputs

The project generates the following files:

| File | Purpose |
|---|---|
| `class_distribution.png` | Visualizes the target class distribution |
| `ad_click_reasons.png` | Visualizes click rates by browsing history |
| `confusion_matrix.png` | Displays the selected model's classification results |
| `roc_curve.png` | Visualizes the ROC curve |
| `prediction_results.csv` | Stores actual and predicted click values |
| `ad_click_model.pkl` | Stores the trained preprocessing and prediction pipeline |

## How to Run the Project

### 1. Navigate to the Project Folder

```bash
cd AD_Click_Data
```

### 2. Install Dependencies

```bash
pip install pandas matplotlib seaborn scikit-learn joblib
```

### 3. Run the Python Script

Ensure the dataset `ad_click_dataset.csv` is available at the path expected by the script.

```bash
python AD_Click_Prediction.py
```

After successful execution, the model, evaluation visualizations, and prediction results will be generated at the paths configured in the script.

## Key Learnings

- Data exploration and missing value analysis
- Numerical and categorical data preprocessing
- Building end-to-end Scikit-learn pipelines
- Handling class imbalance using balanced class weights
- Comparing classification models using multiple metrics
- Evaluating models with a confusion matrix and ROC-AUC
- Saving and reloading trained Machine Learning pipelines

## Conclusion

The project successfully implements an end-to-end advertisement click prediction workflow. Among the evaluated models, **Random Forest (Balanced)** achieved the best Test ROC-AUC of **0.6994**, with an accuracy of **67.85%** and Macro F1 score of **64.25%**.

These results provide a baseline for further experimentation and model improvement. The project also demonstrates model persistence and the generation of evaluation outputs for a Machine Learning portfolio.

*Note: The reported metrics are specific to this dataset split and experiment. They do not guarantee the same performance on new or unseen real-world data.*
