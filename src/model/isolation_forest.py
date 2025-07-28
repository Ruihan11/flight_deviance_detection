
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score   # accuracy_score not really used but imported for consistency

def train(X_train, X_test, y_train, y_test):
    """
    1. Pipeline(StandardScaler + IsolationForest)
    2. 训练后对训练集和测试集分别打标签（1: normal, -1: anomaly）
    3. 打印正常/异常样本数量，保存模型
    """
    print(">> ISO FOREST")
    os.makedirs("models", exist_ok=True)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("iso", IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42
        ))
    ])

    # —— 训练 —— 
    pipe.fit(X_train)

    # —— 训练集 & 测试集预测 —— 
    pred_tr = pipe.predict(X_train)
    pred_te = pipe.predict(X_test)

    tr_norm = (pred_tr == 1).sum()
    tr_anom = (pred_tr == -1).sum()
    te_norm = (pred_te == 1).sum()
    te_anom = (pred_te == -1).sum()

    print(f"Train: normal={tr_norm}/{len(X_train)}, anomalies={tr_anom}/{len(X_train)}")
    print(f"Test:  normal={te_norm}/{len(X_test)},  anomalies={te_anom}/{len(X_test)}")

    # —— 保存模型 —— 
    joblib.dump(pipe, "models/IsolationForest_pipeline.pkl")
