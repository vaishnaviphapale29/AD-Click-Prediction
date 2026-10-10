import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

def AdClickPredict(DataPath):

    Border = "-" * 50

    # Dataset and output file paths
    BASE_DIR = Path(__file__).resolve().parent
    DATASET_PATH = BASE_DIR / DataPath
    IMAGE_DIR = BASE_DIR

    #--------------------------------------------------
    # Step 1 : Load Dataset
    #--------------------------------------------------
    print(Border)
    print("Step 1 : Load Dataset")
    print(Border)

    if not DATASET_PATH.exists():
        print("Dataset not found!")
        print("Expected path:", DATASET_PATH)
        return

    df = pd.read_csv(DATASET_PATH)

    print("First 5 Records:")
    print(df.head())

    print("\nDataset Shape:")
    print(df.shape)

    print("\nDataset Columns:")
    print(df.columns.tolist())

    df.drop(
        columns=["id", "full_name"],
        errors="ignore",
        inplace=True
    )

    #--------------------------------------------------
    # Step 2 : Check Missing Values
    #--------------------------------------------------
    print("\n" + Border)
    print("Step 2 : Check Missing Values")
    print(Border)

    print(df.isnull().sum())

    if "click" not in df.columns:
        print("Error: Target column 'click' not found!")
        return

    df.dropna(subset=["click"], inplace=True)

    if df.empty:
        print("Error: Dataset is empty after removing missing targets.")
        return

    if df["click"].nunique() < 2:
        print("Error: Target must contain at least two classes.")
        return

    print("\nMissing values will be handled by the ML Pipeline.")

    #--------------------------------------------------
    # Step 3 : Statistical Summary
    #--------------------------------------------------
    print("\n" + Border)
    print("Step 3 : Statistical Summary")
    print(Border)

    print(df.describe(include="all"))

    print("\nClick Class Distribution:")
    print(df["click"].value_counts())

    #--------------------------------------------------
    # Step 4 : Click Distribution Visualization
    #--------------------------------------------------
    print("\n" + Border)
    print("Step 4 : Click Distribution Visualization")
    print(Border)

    plt.figure(figsize=(8, 5))

    df["click"].value_counts().sort_index().plot(
        kind="bar",
        color="skyblue",
        edgecolor="black"
    )

    plt.title("Ad Click Distribution")
    plt.xlabel("Click Class")
    plt.ylabel("Number of Records")
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        IMAGE_DIR / "class_distribution.png",
        dpi=300
    )

    plt.show()
    plt.close()

    #--------------------------------------------------
    # Step 4.1 : Why Do People Click Ads?
    #--------------------------------------------------
    print("\n" + Border)
    print("Step 4.1 : Ad Click Reasons Visualization")
    print(Border)

    if "browsing_history" in df.columns:

        ClickData = df.dropna(
            subset=["browsing_history", "click"]
        )

        ClickRate = (
            ClickData.groupby("browsing_history")["click"]
            .mean() * 100
        ).sort_values(ascending=False)

        plt.figure(figsize=(9, 5))

        ClickRate.plot(
            kind="bar",
            color="orange",
            edgecolor="black"
        )

        plt.title("Why Do People Click Ads? - Browsing History")
        plt.xlabel("Browsing History")
        plt.ylabel("Click Rate (%)")
        plt.xticks(rotation=30, ha="right")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            IMAGE_DIR / "ad_click_reasons.png",
            dpi=300
        )

        plt.show()
        plt.close()

        print("Click Rate by Browsing History:")
        print(ClickRate.round(2))

    else:
        print("Browsing history column not found. Graph skipped.")

    #--------------------------------------------------
    # Step 5 : Separate Input and Output
    #--------------------------------------------------
    print("\n" + Border)
    print("Step 5 : Separate Input and Output")
    print(Border)

    X = df.drop(columns=["click"])
    Y = df["click"]

    NumericColumns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    CategoricalColumns = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    print("Input Features:")
    print(X.columns.tolist())

    print("\nTarget Column: click")
    print("Numeric Columns:", NumericColumns)
    print("Categorical Columns:", CategoricalColumns)

    #--------------------------------------------------
    # Step 6 : Train-Test Split
    #--------------------------------------------------
    print("\n" + Border)
    print("Step 6 : Train-Test Split")
    print(Border)

    ClassCounts = Y.value_counts()

    if ClassCounts.min() < 2:
        print("Error: Each class needs at least two records.")
        return

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.20,
        random_state=42,
        stratify=Y
    )

    print("Training Records:", X_train.shape[0])
    print("Testing Records:", X_test.shape[0])

    #--------------------------------------------------
    # Step 7 : Create ML Pipeline
    #--------------------------------------------------
    print("\n" + Border)
    print("Step 7 : Create ML Pipeline")
    print(Border)

    NumericPipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    CategoricalPipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    Preprocessor = ColumnTransformer([
        ("numeric", NumericPipeline, NumericColumns),
        ("categorical", CategoricalPipeline, CategoricalColumns)
    ])

    ModelPipeline = Pipeline([
        ("preprocessor", Preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ])

    print("ML Pipeline created successfully.")

    #--------------------------------------------------
    # Step 8 : Train Model
    #--------------------------------------------------
    print("\n" + Border)
    print("Step 8 : Train Logistic Regression Model")
    print(Border)

    ModelPipeline.fit(X_train, Y_train)

    print("Model trained successfully.")

    #--------------------------------------------------
    # Step 9 : Make Predictions
    #--------------------------------------------------
    print("\n" + Border)
    print("Step 9 : Make Predictions")
    print(Border)

    Y_pred = ModelPipeline.predict(X_test)

    Results = pd.DataFrame({
        "Actual Click": Y_test.to_numpy(),
        "Predicted Click": Y_pred
    })

    print(Results.head(20))

    Results.to_csv(
        IMAGE_DIR / "prediction_results.csv",
        index=False
    )

    #--------------------------------------------------
    # Step 10 : Model Evaluation
    #--------------------------------------------------
    print("\n" + Border)
    print("Step 10 : Model Evaluation")
    print(Border)

    Accuracy = accuracy_score(Y_test, Y_pred)

    Precision = precision_score(
        Y_test,
        Y_pred,
        average="weighted",
        zero_division=0
    )

    Recall = recall_score(
        Y_test,
        Y_pred,
        average="weighted",
        zero_division=0
    )

    F1 = f1_score(
        Y_test,
        Y_pred,
        average="weighted",
        zero_division=0
    )

    print("Accuracy :", round(Accuracy, 4))
    print("Precision:", round(Precision, 4))
    print("Recall   :", round(Recall, 4))
    print("F1 Score :", round(F1, 4))

    print("\nClassification Report:")
    print(classification_report(
        Y_test,
        Y_pred,
        zero_division=0
    ))

    #--------------------------------------------------
    # Step 11 : Confusion Matrix
    #--------------------------------------------------
    print("\n" + Border)
    print("Step 11 : Confusion Matrix")
    print(Border)

    CM = confusion_matrix(
        Y_test,
        Y_pred,
        labels=ModelPipeline.classes_
    )

    print("Confusion Matrix:")
    print(CM)

    Display = ConfusionMatrixDisplay(
        confusion_matrix=CM,
        display_labels=ModelPipeline.classes_
    )

    Display.plot(
        cmap="Blues",
        values_format="d"
    )

    plt.title("Ad Click Prediction - Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        IMAGE_DIR / "confusion_matrix.png",
        dpi=300
    )

    plt.show()
    plt.close()

    #--------------------------------------------------
    # Step 12 : ROC Curve
    #--------------------------------------------------
    print("\n" + Border)
    print("Step 12 : ROC Curve")
    print(Border)

    Classes = ModelPipeline.classes_

    if len(Classes) == 2:

        PositiveClass = Classes[1]

        Y_test_binary = (
            Y_test == PositiveClass
        ).astype(int)

        Y_probability = ModelPipeline.predict_proba(
            X_test
        )[:, 1]

        FPR, TPR, Thresholds = roc_curve(
            Y_test_binary,
            Y_probability
        )

        ROC_AUC = auc(FPR, TPR)

        print("ROC-AUC Score:", round(ROC_AUC, 4))

        plt.figure(figsize=(8, 5))

        plt.plot(
            FPR,
            TPR,
            label=f"ROC Curve (AUC = {ROC_AUC:.3f})"
        )

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            label="Random Classifier"
        )

        plt.title("ROC Curve - Ad Click Prediction")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            IMAGE_DIR / "roc_curve.png",
            dpi=300
        )

        plt.show()
        plt.close()

    else:
        print("ROC curve skipped: target has more than two classes.")

    #--------------------------------------------------
    # Step 13 : Cross Validation
    #--------------------------------------------------
    print("\n" + Border)
    print("Step 13 : Cross Validation")
    print(Border)

    MinimumClassCount = int(
        Y_train.value_counts().min()
    )

    CV = min(5, MinimumClassCount)

    if CV >= 2:

        CV_Scores = cross_val_score(
            ModelPipeline,
            X_train,
            Y_train,
            cv=CV,
            scoring="accuracy"
        )

        print("Cross Validation Scores:", CV_Scores)
        print("Mean CV Accuracy:", round(CV_Scores.mean(), 4))

    else:
        print("Cross-validation skipped: insufficient records.")

    #--------------------------------------------------
    # Step 14 : Save Model
    #--------------------------------------------------
    print("\n" + Border)
    print("Step 14 : Save Model")
    print(Border)

    ModelPath = IMAGE_DIR / "ad_click_model.pkl"

    joblib.dump(
        ModelPipeline,
        ModelPath
    )

    print("Model saved successfully at:", ModelPath)

    #--------------------------------------------------
    # Step 15 : Load Saved Model
    #--------------------------------------------------
    print("\n" + Border)
    print("Step 15 : Load Saved Model")
    print(Border)

    LoadedModel = joblib.load(ModelPath)

    print("Saved model loaded successfully!")

    #--------------------------------------------------
    # Step 16 : Test Loaded Model
    #--------------------------------------------------
    print("\n" + Border)
    print("Step 16 : Test Loaded Model")
    print(Border)

    LoadedPredictions = LoadedModel.predict(X_test)

    print("Actual Values:")
    print(Y_test.to_numpy()[:10])

    print("\nPredictions from Loaded Model:")
    print(LoadedPredictions[:10])

    LoadedAccuracy = accuracy_score(
        Y_test,
        LoadedPredictions
    )

    print("Loaded Model Accuracy:", round(LoadedAccuracy, 4))

    #--------------------------------------------------
    # Step 17 : Final Summary
    #--------------------------------------------------
    print("\n" + Border)
    print("Step 17 : Final Summary")
    print(Border)

    print("Dataset Records:", df.shape[0])
    print("Total Features:", X.shape[1])
    print("Training Records:", X_train.shape[0])
    print("Testing Records:", X_test.shape[0])
    print("Model: Logistic Regression")
    print("Accuracy:", round(Accuracy, 4))

    print("\nGenerated Files:")
    print("1. class_distribution.png")
    print("2. ad_click_reasons.png")
    print("3. confusion_matrix.png")

    if len(Classes) == 2:
        print("4. roc_curve.png")
        print("5. prediction_results.csv")
        print("6. ad_click_model.pkl")
    else:
        print("4. prediction_results.csv")
        print("5. ad_click_model.pkl")

    print("\nAd Click Prediction Completed Successfully!")

def main():
    AdClickPredict("ad_click_dataset.csv")

if __name__ == "__main__":
    main()