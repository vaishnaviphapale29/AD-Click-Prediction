import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path

from sklearn.base import clone
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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

    print("\nClick Class Percentage:")
    print((df["click"].value_counts(normalize=True) * 100).round(2))

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
    # Step 7 : Create Preprocessing Pipeline
    #--------------------------------------------------
    print("\n" + Border)
    print("Step 7 : Create Preprocessing Pipeline")
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

    print("Preprocessing pipeline created successfully.")

    #--------------------------------------------------
    # Step 8 : Create Machine Learning Models
    #--------------------------------------------------
    print("\n" + Border)
    print("Step 8 : Create Machine Learning Models")
    print(Border)

    # class_weight="balanced" stops the model from always predicting
    # the majority class (click = 1)
    Models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000
        ),
        "Logistic Regression (Balanced)": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),
        "Random Forest (Balanced)": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    }

    for ModelName in Models:
        print("-", ModelName)

    #--------------------------------------------------
    # Step 9 : Train and Compare Models
    #--------------------------------------------------
    print("\n" + Border)
    print("Step 9 : Train and Compare Models")
    print(Border)

    MinimumClassCount = int(
        Y_train.value_counts().min()
    )

    CV = min(5, MinimumClassCount)

    FittedModels = {}
    ComparisonRows = []

    for ModelName, Model in Models.items():

        print("Training:", ModelName)

        Pipe = Pipeline([
            ("preprocessor", clone(Preprocessor)),
            ("model", Model)
        ])

        Pipe.fit(X_train, Y_train)

        Pred = Pipe.predict(X_test)

        if CV >= 2:
            CV_AUC = cross_val_score(
                Pipe,
                X_train,
                Y_train,
                cv=CV,
                scoring="roc_auc"
            ).mean()
        else:
            CV_AUC = float("nan")

        TestAUC = None

        if len(Pipe.classes_) == 2:
            Probability = Pipe.predict_proba(X_test)[:, 1]
            FPR_, TPR_, _ = roc_curve(
                (Y_test == Pipe.classes_[1]).astype(int),
                Probability
            )
            TestAUC = auc(FPR_, TPR_)

        ComparisonRows.append({
            "Model": ModelName,
            "Accuracy": accuracy_score(Y_test, Pred),
            "Macro_F1": f1_score(
                Y_test, Pred,
                average="macro",
                zero_division=0
            ),
            "Recall_Class_0": recall_score(
                Y_test, Pred,
                labels=[Pipe.classes_[0]],
                average="macro",
                zero_division=0
            ),
            "Test_ROC_AUC": TestAUC,
            "CV_ROC_AUC": CV_AUC
        })

        FittedModels[ModelName] = Pipe

    ComparisonTable = pd.DataFrame(ComparisonRows).round(4)

    print("\nModel Comparison:")
    print(ComparisonTable.to_string(index=False))

    #--------------------------------------------------
    # Step 10 : Select Best Model and Predict
    #--------------------------------------------------
    print("\n" + Border)
    print("Step 10 : Select Best Model and Make Predictions")
    print(Border)

    SortColumn = "Test_ROC_AUC"

    if ComparisonTable[SortColumn].isnull().all():
        SortColumn = "Macro_F1"

    BestModelName = ComparisonTable.sort_values(
        SortColumn,
        ascending=False
    ).iloc[0]["Model"]

    ModelPipeline = FittedModels[BestModelName]

    print("Best Model (by", SortColumn + "):", BestModelName)

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
    # Step 11 : Model Evaluation
    #--------------------------------------------------
    print("\n" + Border)
    print("Step 11 : Model Evaluation")
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

    MacroF1 = f1_score(
        Y_test,
        Y_pred,
        average="macro",
        zero_division=0
    )

    print("Accuracy :", round(Accuracy, 4))
    print("Precision:", round(Precision, 4))
    print("Recall   :", round(Recall, 4))
    print("F1 Score :", round(F1, 4))
    print("Macro F1 :", round(MacroF1, 4))

    print("\nClassification Report:")
    print(classification_report(
        Y_test,
        Y_pred,
        zero_division=0
    ))

    #--------------------------------------------------
    # Step 12 : Confusion Matrix
    #--------------------------------------------------
    print("\n" + Border)
    print("Step 12 : Confusion Matrix")
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
    # Step 13 : ROC Curve
    #--------------------------------------------------
    print("\n" + Border)
    print("Step 13 : ROC Curve")
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
    # Step 15 : Load and Test Saved Model
    #--------------------------------------------------
    print("\n" + Border)
    print("Step 15 : Load and Test Saved Model")
    print(Border)

    LoadedModel = joblib.load(ModelPath)

    print("Saved model loaded successfully!")

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
    # Step 16 : Final Summary
    #--------------------------------------------------
    print("\n" + Border)
    print("Step 16 : Final Summary")
    print(Border)

    print("Dataset Records:", df.shape[0])
    print("Total Features:", X.shape[1])
    print("Training Records:", X_train.shape[0])
    print("Testing Records:", X_test.shape[0])
    print("Best Model:", BestModelName)
    print("Accuracy:", round(Accuracy, 4))
    print("Macro F1:", round(MacroF1, 4))

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