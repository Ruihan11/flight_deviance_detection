
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train(X_train, X_test, y_train, y_test):
    """
    1. Pipeline(StandardScaler + MLPClassifier)
    2. 5‑折交叉验证评估
    3. 在测试集上评估并保存模型
    """
    print(">> MLP")
    os.makedirs("models", exist_ok=True)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=1e-4,
            learning_rate='adaptive',
            max_iter=1000,
            random_state=42,
            verbose=False
        ))
    ])

    # —— 交叉验证 —— 
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=kf, scoring='accuracy', n_jobs=-1)
    print(f"MLP Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # —— 全量训练 & 测试集评估 —— 
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print(f"\nMLP Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # —— 保存模型 —— 
    joblib.dump(pipe, "models/MLP_pipeline.pkl")
