
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train(X_train, X_test, y_train, y_test):
    """
    1. Pipeline(StandardScaler + LGBMClassifier)
    2. 通过 verbose=-1 彻底关闭 LightGBM Info 日志
    3. 在测试集上评估并打印指标
    4. 保存整个 Pipeline
    """
    print(">> LIGHTGBM")
    
    os.makedirs("models", exist_ok=True)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lgb", LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=-1,
            num_leaves=31,
            random_state=42,
            verbose=-1    # 关闭 Info 日志
        ))
    ])

    # 训练
    pipe.fit(X_train, y_train)

    # 预测
    y_pred = pipe.predict(X_test)

    # 评估
    print(f"\nLightGBM Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # 保存模型
    joblib.dump(pipe, "models/LGBMClassifier_pipeline.pkl")
