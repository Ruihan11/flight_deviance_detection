import os
import joblib

def train(X_train, X_test, y_train, y_test):
    """
    GPU-accelerated SVM with RAPIDS cuML.
    1. 动态导入 cudf & cuML，若未安装则跳过。
    2. Pandas → cuDF → cuML SVC → 预测 → 转回 NumPy
    3. 打印 accuracy，并将模型存为 pickle。
    """
    try:
        import cudf
        from cuml.svm import SVC as cuSVC
    except ImportError:
        print("cuML / cuDF 未安装，跳过 cuML GPU SVC")
        return

    os.makedirs("models", exist_ok=True)

    # 转 cuDF
    Xc_train = cudf.DataFrame.from_pandas(X_train)
    yc_train = cudf.Series(y_train.values)
    Xc_test  = cudf.DataFrame.from_pandas(X_test)

    # 构造并训练 GPU SVC（RBF 核）
    svc_gpu = cuSVC(kernel='rbf', C=1.0, gamma='scale')
    svc_gpu.fit(Xc_train, yc_train)

    # 预测并转回 NumPy
    preds = svc_gpu.predict(Xc_test).to_pandas().values
    acc = (preds == y_test.values).mean()
    print(f"\ncuML SVC GPU Test Accuracy: {acc:.4f}")

    # 保存模型
    joblib.dump(svc_gpu, "models/cuml_svc.pkl")