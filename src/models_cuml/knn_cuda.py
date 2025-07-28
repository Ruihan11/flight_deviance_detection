import os
import joblib

def train(X_train, X_test, y_train, y_test):
    """
    GPU-accelerated KNN with RAPIDS cuML.
    1. 动态导入 cudf & cuML，若未安装则跳过。
    2. Pandas → cuDF → cuML KNeighborsClassifier → 预测 → 转回 NumPy
    3. 打印 accuracy，并将模型存为 pickle。
    """
    try:
        import cudf
        from cuml.neighbors import KNeighborsClassifier as cuKNN
    except ImportError:
        print("cuML / cuDF 未安装，跳过 cuML GPU KNN")
        return

    os.makedirs("models", exist_ok=True)

    # 转成 cuDF 格式
    Xc_train = cudf.DataFrame.from_pandas(X_train)
    yc_train = cudf.Series(y_train.values)
    Xc_test  = cudf.DataFrame.from_pandas(X_test)

    # 构造并训练 GPU KNN
    knn_gpu = cuKNN(n_neighbors=10)
    knn_gpu.fit(Xc_train, yc_train)

    # 预测并转回 NumPy
    preds = knn_gpu.predict(Xc_test).to_pandas().values
    acc = (preds == y_test.values).mean()
    print(f"\ncuML KNN GPU Test Accuracy: {acc:.4f}")

    # 保存模型
    joblib.dump(knn_gpu, "models/cuml_knn.pkl")