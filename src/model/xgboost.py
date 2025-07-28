
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

def train(X_train, X_test, y_train, y_test):
    """
    1. Pipeline(StandardScaler + XGBClassifier)
    2. 去除已弃用的 use_label_encoder 参数，使用 verbosity=0 控制日志
    3. 在测试集上评估并打印指标
    4. 保存整个 Pipeline
    """
    print(">> XGBOOST")
    
    os.makedirs("models", exist_ok=True)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            verbosity=0,
            random_state=42
        ))
    ])

    # 静默训练过程中可能的轻量警告
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X_train, y_train)

    # 测试集评估
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*feature_names.*")
        y_pred = pipe.predict(X_test)

    print(f"\nXGBoost Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # 保存模型
    joblib.dump(pipe, "models/XGBClassifier_pipeline.pkl")
