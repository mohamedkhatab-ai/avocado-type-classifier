import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from warnings import filterwarnings

filterwarnings("ignore")


# -------------------------------------------------------------
# 1. Neural Network Architecture Definition
# -------------------------------------------------------------
class TypeClassifierNN(nn.Module):
    """Multi-Layer Perceptron (MLP) for Binary Type Classification."""

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# -------------------------------------------------------------
# 2. Data Loading & Feature Engineering Function
# -------------------------------------------------------------
def load_and_preprocess_data(filepath="avocado.csv"):
    """Loads CSV, extracts date features, encodes categorical variables, and splits data."""
    df = pd.read_csv(filepath)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    # Extract date features
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df = df.drop(columns=["Date"])

    # Categorical encodings
    le_region = LabelEncoder()
    df["region_code"] = le_region.fit_transform(df["region"])

    le_type = LabelEncoder()
    df["type_code"] = le_type.fit_transform(df["type"])  # 0: conventional, 1: organic

    features = [
        "AveragePrice",
        "Total Volume",
        "4046",
        "4225",
        "4770",
        "Total Bags",
        "Small Bags",
        "Large Bags",
        "XLarge Bags",
        "year",
        "Month",
        "Day",
        "region_code",
    ]

    X = df[features]
    y = df["type_code"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return (X_train, X_test, y_train, y_test), le_type.classes_


# -------------------------------------------------------------
# 3. AdaBoost Training & Evaluation Function
# -------------------------------------------------------------
def train_adaboost_model(X_train, y_train, X_test, y_test, class_names):
    """Trains and evaluates the AdaBoost Classifier."""
    ada_clf = AdaBoostClassifier(n_estimators=100, random_state=42, algorithm="SAMME")
    ada_clf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, ada_clf.predict(X_train))
    test_acc = accuracy_score(y_test, ada_clf.predict(X_test))

    print("--- AdaBoost Classifier ---")
    print(f"Train Accuracy: {train_acc * 100:.2f}%")
    print(f"Test Accuracy:  {test_acc * 100:.2f}%\n")
    print(classification_report(y_test, ada_clf.predict(X_test), target_names=class_names))

    return ada_clf


# -------------------------------------------------------------
# 4. Deep Learning Training & Evaluation Function
# -------------------------------------------------------------
def train_deep_learning_model(
    X_train, y_train, X_test, y_test, epochs=30, batch_size=64, lr=0.005
):
    """Scales data, trains PyTorch MLP model, and prints accuracy metrics."""
    # Scale features
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Convert to Tensors
    X_train_t = torch.tensor(X_train_sc, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test_sc, dtype=torch.float32)

    # Model Setup
    torch.manual_seed(42)
    model = TypeClassifierNN(X_train_t.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training Loop
    loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True
    )
    for epoch in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        train_pred = (model(X_train_t).numpy() > 0.5).astype(int)
        test_pred = (model(X_test_t).numpy() > 0.5).astype(int)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print("--- Deep Learning (MLP) Classifier ---")
    print(f"Train Accuracy: {train_acc * 100:.2f}%")
    print(f"Test Accuracy:  {test_acc * 100:.2f}%")

    return model, scaler


# -------------------------------------------------------------
# 5. Main Pipeline Execution
# -------------------------------------------------------------
if __name__ == "__main__":
    # Load and split
    (X_train, X_test, y_train, y_test), target_classes = load_and_preprocess_data(
        "avocado.csv"
    )

    # Train AdaBoost
    ada_model = train_adaboost_model(X_train, y_train, X_test, y_test, target_classes)

    # Train Deep Learning MLP
    dl_model, feature_scaler = train_deep_learning_model(
        X_train, y_train, X_test, y_test
    )