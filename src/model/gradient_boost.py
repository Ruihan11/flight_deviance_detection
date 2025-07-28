
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train(X_train, X_test, y_train, y_test):
    """
    1. Pipeline(StandardScaler + GradientBoostingClassifier)
    2. 在测试集上评估并打印指标
    3. 保存整个 Pipeline
    """
    print(">> GRADIENT BOOST")

    os.makedirs("models", exist_ok=True)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("gb", GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        ))
    ])

    # 训练
    pipe.fit(X_train, y_train)

    # 测试集评估
    y_pred = pipe.predict(X_test)
    print(f"\nGradient Boosting Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # 保存模型
    joblib.dump(pipe, "models/GradientBoosting_pipeline.pkl")
