import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten

DATA_PATH = "data/Titanic-Dataset.csv"
MODEL_PATH = "models/titanic_best_model.pkl"

os.makedirs("models", exist_ok=True)

print("\n" + "=" * 55)
print("🚢 TITANIC SURVIVAL PREDICTION")
print("MODEL TRAINING STARTED")
print("=" * 55)

df = pd.read_csv(DATA_PATH)

print("\n✅ Dataset loaded successfully!")
print("Dataset Shape:", df.shape)

print("\n📋 First 5 rows:")
print(df.head())

print("\n🔍 Missing Values:")
print(df.isnull().sum())

print("\n🎯 Survival Distribution:")
print(df["Survived"].value_counts())

features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]

X = df[features]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n📊 Training Samples:", len(X_train))
print("📊 Testing Samples:", len(X_test))

numeric_features = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]

categorical_features = [
    "Sex",
    "Embarked"
]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ]
)

decision_tree = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", DecisionTreeClassifier(
            random_state=42,
            max_depth=5
        ))
    ]
)

random_forest = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=300,
            random_state=42
        ))
    ]
)

gradient_boosting = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            random_state=42
        ))
    ]
)

xgb_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        ))
    ]
)

ml_models = {
    "Logistic Regression": logistic_model,
    "Decision Tree": decision_tree,
    "Random Forest": random_forest,
    "Gradient Boosting": gradient_boosting,
    "XGBoost": xgb_model
}

ml_results = []

for name, model in ml_models.items():

    print("\n🔄 Training:", name)

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    result = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "F1 Score": f1_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "ROC-AUC": roc_auc_score(
            y_test,
            y_prob
        )
    }

    ml_results.append(result)

    print(
        f"   Accuracy: {result['Accuracy']:.4f}"
    )

    print(
        f"   F1 Score: {result['F1 Score']:.4f}"
    )

    print(
        f"   ROC-AUC: {result['ROC-AUC']:.4f}"
    )

ml_results_df = pd.DataFrame(
    ml_results
)

print("\n" + "-" * 55)
print("📊 MACHINE LEARNING RESULTS")
print("-" * 55)

print(
    ml_results_df.to_string(
        index=False
    )
)

dl_numeric = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

dl_categorical = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

dl_preprocessor = ColumnTransformer(
    [
        ("num", dl_numeric, numeric_features),
        ("cat", dl_categorical, categorical_features)
    ]
)

X_train_dl = dl_preprocessor.fit_transform(
    X_train
)

X_test_dl = dl_preprocessor.transform(
    X_test
)

X_train_dl = np.asarray(
    X_train_dl
).astype("float32")

X_test_dl = np.asarray(
    X_test_dl
).astype("float32")

print("\n🧠 Training ANN model...")

ann_model = Sequential(
    [
        Dense(
            64,
            activation="relu",
            input_shape=(X_train_dl.shape[1],)
        ),

        Dropout(0.30),

        Dense(
            32,
            activation="relu"
        ),

        Dropout(0.20),

        Dense(
            16,
            activation="relu"
        ),

        Dense(
            1,
            activation="sigmoid"
        )
    ]
)

ann_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

ann_model.fit(
    X_train_dl,
    y_train,
    validation_split=0.20,
    epochs=50,
    batch_size=32,
    verbose=0
)

ann_prob = ann_model.predict(
    X_test_dl,
    verbose=0
).ravel()

ann_pred = (
    ann_prob >= 0.5
).astype(int)

ann_results = {
    "Model": "ANN",
    "Accuracy": accuracy_score(
        y_test,
        ann_pred
    ),
    "Precision": precision_score(
        y_test,
        ann_pred,
        zero_division=0
    ),
    "Recall": recall_score(
        y_test,
        ann_pred,
        zero_division=0
    ),
    "F1 Score": f1_score(
        y_test,
        ann_pred,
        zero_division=0
    ),
    "ROC-AUC": roc_auc_score(
        y_test,
        ann_prob
    )
}

print("\n🧠 Training 1D CNN model...")

X_train_cnn = X_train_dl.reshape(
    X_train_dl.shape[0],
    X_train_dl.shape[1],
    1
)

X_test_cnn = X_test_dl.reshape(
    X_test_dl.shape[0],
    X_test_dl.shape[1],
    1
)

cnn_model = Sequential(
    [
        Conv1D(
            32,
            kernel_size=3,
            activation="relu",
            input_shape=(
                X_train_cnn.shape[1],
                1
            )
        ),

        MaxPooling1D(
            pool_size=2
        ),

        Conv1D(
            64,
            kernel_size=3,
            activation="relu"
        ),

        Flatten(),

        Dense(
            32,
            activation="relu"
        ),

        Dropout(0.30),

        Dense(
            1,
            activation="sigmoid"
        )
    ]
)

cnn_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

cnn_model.fit(
    X_train_cnn,
    y_train,
    validation_split=0.20,
    epochs=50,
    batch_size=32,
    verbose=0
)

cnn_prob = cnn_model.predict(
    X_test_cnn,
    verbose=0
).ravel()

cnn_pred = (
    cnn_prob >= 0.5
).astype(int)


cnn_results = {
    "Model": "1D CNN",
    "Accuracy": accuracy_score(
        y_test,
        cnn_pred
    ),
    "Precision": precision_score(
        y_test,
        cnn_pred,
        zero_division=0
    ),
    "Recall": recall_score(
        y_test,
        cnn_pred,
        zero_division=0
    ),
    "F1 Score": f1_score(
        y_test,
        cnn_pred,
        zero_division=0
    ),
    "ROC-AUC": roc_auc_score(
        y_test,
        cnn_prob
    )
}

dl_results_df = pd.DataFrame(
    [
        ann_results,
        cnn_results
    ]
)

all_results = pd.concat(
    [
        ml_results_df,
        dl_results_df
    ],
    ignore_index=True
)

all_results = all_results.sort_values(
    by="F1 Score",
    ascending=False
).reset_index(drop=True)

print("\n" + "=" * 55)
print("📊 ALL MODEL RESULTS")
print("=" * 55)

print(
    all_results.to_string(
        index=False
    )
)

all_results.to_csv(
    "model_results.csv",
    index=False
)

print("\n✅ Model results saved to model_results.csv")

best_ml_name = (
    ml_results_df
    .sort_values(
        by="F1 Score",
        ascending=False
    )
    .iloc[0]["Model"]
)

print("\n🏆 Best Machine Learning Model:")
print(best_ml_name)

best_ml_model = ml_models[
    best_ml_name
]

joblib.dump(
    best_ml_model,
    MODEL_PATH
)

print(
    f"\n✅ Best model saved successfully!"
)

print(
    f"📁 Location: {MODEL_PATH}"
)

best_prediction = best_ml_model.predict(
    X_test
)

cm = confusion_matrix(
    y_test,
    best_prediction
)

print("\n📌 Confusion Matrix:")
print(cm)

print("\n📋 Classification Report:")

print(
    classification_report(
        y_test,
        best_prediction,
        zero_division=0
    )
)

plt.figure(
    figsize=(14, 7)
)

all_results.set_index(
    "Model"
)[
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]
].plot(
    kind="bar"
)

plt.title(
    "🚢 Titanic Model Performance Comparison"
)

plt.ylabel(
    "Score"
)

plt.xlabel(
    "Model"
)

plt.ylim(
    0,
    1
)

plt.xticks(
    rotation=30
)

plt.legend(
    loc="lower right"
)

plt.tight_layout()

plt.savefig(
    "model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\n📈 Model comparison graph saved!"
)

plt.figure(
    figsize=(6, 5)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Survived", "Survived"],
    yticklabels=["Not Survived", "Survived"]
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.title(
    "Titanic Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "📊 Confusion matrix graph saved!"
)

print("\n" + "=" * 55)
print("🎉 PROJECT TRAINING COMPLETED!")
print("=" * 55)

print(
    f"Best ML Model: {best_ml_name}"
)

print(
    f"Model File: {MODEL_PATH}"
)

print(
    "Results File: model_results.csv"
)

print(
    "Comparison Graph: model_comparison.png"
)

print(
    "Confusion Matrix: confusion_matrix.png"
)

print("=" * 55)