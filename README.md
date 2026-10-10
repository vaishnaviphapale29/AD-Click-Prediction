# Ad Click Prediction using Logistic Regression

## Project Overview

The **Ad Click Prediction** project uses Machine Learning to predict whether a user will click on an advertisement based on features such as age, gender, device type, ad position, browsing history, and time of day.

This project uses **Logistic Regression** with a complete Scikit-learn Pipeline for data preprocessing, model training, prediction, and evaluation.

## Objectives

- Predict whether a user will click an advertisement.
- Handle missing values using imputation techniques.
- Encode categorical features into a machine-readable format.
- Scale numerical features.
- Evaluate model performance using multiple metrics.
- Visualize ad click patterns using graphs.
- Save and reload the trained Machine Learning model.

## Project Structure

```text
Ad-Click-Prediction/
│
├── AdClickPrediction.py
├── ad_click_dataset.csv
├── README.md
├── requirements.txt
│
├── class_distribution.png
├── ad_click_reasons.png
├── confusion_matrix.png
├── roc_curve.png
│
├── prediction_results.csv
└── ad_click_model.pkl
```

*Note: The PNG images, prediction CSV, and model file are generated when the program runs successfully.*

## Dataset Information

The dataset contains information about users and their interactions with advertisements.

| Feature | Description |
|---|---|
| `id` | Unique record identifier |
| `full_name` | User name |
| `age` | User's age |
| `gender` | User's gender |
| `device_type` | Device used by the user |
| `ad_position` | Advertisement placement |
| `browsing_history` | User's browsing category |
| `time_of_day` | Time when the user browses |
| `click` | Target variable: click or no click |

The `id` and `full_name` columns are removed before model training. The `click` column is the target variable.

## Technologies Used

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Logistic Regression
- Machine Learning Pipeline

## Machine Learning Workflow

1. Load the dataset.
2. Inspect missing values and dataset statistics.
3. Visualize the click distribution.
4. Analyze click rates by browsing history.
5. Separate input features and target variable.
6. Split the dataset into training and testing sets.
7. Handle missing numerical values using median imputation.
8. Handle missing categorical values using most-frequent imputation.
9. Scale numerical features using StandardScaler.
10. Encode categorical features using OneHotEncoder.
11. Train the Logistic Regression model.
12. Predict advertisement clicks.
13. Evaluate model performance.
14. Perform cross-validation.
15. Save and reload the trained model.

## Visualizations

### 1. Ad Click Distribution

Displays the number of records for each click class.

File: `class_distribution.png`

### 2. Ad Click Rate by Browsing History

Compares the percentage of clicks across browsing categories to identify which categories have higher observed click rates.

File: `ad_click_reasons.png`

### 3. Confusion Matrix

Shows the comparison between actual and predicted classes.

File: `confusion_matrix.png`

### 4. ROC Curve

Evaluates the model's ability to distinguish between the two target classes using the ROC curve and AUC score.

File: `roc_curve.png`

## Model Evaluation

The model is evaluated using the following metrics:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted clicks that are correct.
- **Recall:** Proportion of actual clicks correctly identified.
- **F1 Score:** Harmonic mean of precision and recall.
- **Confusion Matrix:** Summarizes correct and incorrect predictions.
- **ROC-AUC:** Measures the model's ability to distinguish between the two classes.
- **Cross-Validation:** Evaluates performance across multiple training folds.

Actual metric values are displayed when the program runs; they depend on the dataset and train-test split.

## Installation and Execution

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual GitHub repository URL.

### Step 2: Open the Project Folder

```bash
cd Ad-Click-Prediction
```

Use your actual folder or repository name if it differs.

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Project

```bash
python AdClickPrediction.py
```

Keep `ad_click_dataset.csv` in the same folder as the Python script unless you change the dataset path in the code.

## Generated Outputs

After successful execution, the project generates:

- `class_distribution.png`
- `ad_click_reasons.png`
- `confusion_matrix.png`
- `roc_curve.png` (for a binary target)
- `prediction_results.csv`
- `ad_click_model.pkl`

The saved model can be loaded later using Joblib for further predictions.

## Future Improvements

- Compare Logistic Regression with Random Forest and other classification algorithms.
- Tune hyperparameters to improve performance.
- Add an interactive prediction interface using Streamlit.
- Perform feature importance and coefficient analysis.
- Monitor model performance on new advertisement data.

## Limitations

The model identifies patterns in the available dataset and does not establish why a user clicks an advertisement. Model performance depends on data quality, feature quality, and class distribution.