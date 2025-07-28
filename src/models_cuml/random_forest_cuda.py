import os
import joblib

def train(X_train, X_test, y_train, y_test):
    """
    GPU 版随机森林，基于 RAPIDS cuML。
    1. 动态导入 cudf & cuML，若未安装则跳过。
    2. Pandas → cuDF → cuML RandomForestClassifier → 预测 → 转回 NumPy
    3. 打印 accuracy，并将模型存为 pickle。
    """
    try:
        import cudf
        from cuml.ensemble import RandomForestClassifier as cuRF
    except ImportError:
        print("cuML / cuDF 未安装，跳过 cuML GPU RandomForest")
        return

    os.makedirs("models", exist_ok=True)

    # 1) 转 cuDF
    Xc_train = cudf.DataFrame.from_pandas(X_train)
    yc_train = cudf.Series(y_train.values)
    Xc_test  = cudf.DataFrame.from_pandas(X_test)

    # 2) 构造 & 训练
    rf_gpu = cuRF(
        n_estimators=200,
        max_depth=20,
        random_state=42
    )
    rf_gpu.fit(Xc_train, yc_train)

    # 3) 预测 & 评估
    preds_rf = rf_gpu.predict(Xc_test).to_pandas().values
    acc = (preds_rf == y_test.values).mean()
    print(f"\ncuML RandomForest GPU Test Accuracy: {acc:.4f}")

    # 4) 保存模型
    joblib.dump(rf_gpu, "models/cuml_random_forest.pkl")