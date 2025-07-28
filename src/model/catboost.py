import os
import joblib
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train(X_train, X_test, y_train, y_test):
    """
    1. 直接训练 CatBoostClassifier
    2. 在测试集上评估并打印指标
    3. 保存模型
    """
    print(">> CATBOOST")
    
    os.makedirs("models", exist_ok=True)

    model = CatBoostClassifier(
        iterations=200,
        learning_rate=0.1,
        depth=6,
        verbose=0,
        random_state=42
    )

    # 训练
    model.fit(X_train, y_train)

    # 测试集评估
    y_pred = model.predict(X_test)
    print(f"\nCatBoost Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # 保存模型
    joblib.dump(model, "models/CatBoostClassifier.pkl")
