
import os
import warnings
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope

def train(X_train, X_test, y_train, y_test):
    """
    1. Pipeline(StandardScaler + EllipticEnvelope)
    2. 提升 support_fraction 到 0.75，避免求逆矩阵收敛问题
    3. 暂时屏蔽拟合时的 'Determinant has increased' 警告
    4. 打印正常/异常 样本数量，保存模型
    """
    print(">> ELLIPTIC ENVELOPE")

    os.makedirs("models", exist_ok=True)

    # 构造 Pipeline
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ell", EllipticEnvelope(
            contamination=0.05,
            support_fraction=0.75,   # 从默认 0.501 提升到 0.75
            random_state=42
        ))
    ])

    # 临时屏蔽 sklearn 对 Determinant 增长的警告
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Determinant has increased; this should not happen"
        )
        pipe.fit(X_train)

    # 对训练集 & 测试集做预测
    pred_tr = pipe.predict(X_train)
    pred_te = pipe.predict(X_test)

    tr_norm = (pred_tr == 1).sum()
    tr_anom = (pred_tr == -1).sum()
    te_norm = (pred_te == 1).sum()
    te_anom = (pred_te == -1).sum()

    print(f"Train: normal={tr_norm}/{len(X_train)}, outliers={tr_anom}/{len(X_train)}")
    print(f"Test:  normal={te_norm}/{len(X_test)},  outliers={te_anom}/{len(X_test)}")

    # 保存模型
    joblib.dump(pipe, "models/EllipticEnvelope_pipeline.pkl")
