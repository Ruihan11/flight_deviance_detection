#!/usr/bin/env python3

from src.data_processing          import load_and_split_data
from src.models.random_forest     import train as train_rf
from src.models.knn               import train as train_knn
from src.models.svm               import train as train_svm
from src.models.mlp               import train as train_mlp
from src.models.isolation_forest  import train as train_iso
from src.models.elliptic_envelope import train as train_ell
from src.models.gradient_boost    import train as train_gb
from src.models.xgboost           import train as train_xgb
from src.models.catboost          import train as train_cb
from src.models.lightgbm          import train as train_lgb

# GPU models
from src.models_cuml.random_forest_cuda     import train as train_rf_cuda
from src.models_cuml.knn_cuda               import train as train_knn_cuda
from src.models_cuml.svc_cuda               import train as train_svc_cuda


def main():
    # —— 1. 数据加载与预处理 —— 
    X_train, X_test, y_train, y_test = load_and_split_data(
        planes_csv="data/plane_features_reduced.csv",
        labels_csv="data/aircraftDatabase.csv",
        test_size=0.2,
        random_state=42
    )

    # —— 2. 依次训练各模型 —— 
    # train_rf(X_train, X_test, y_train, y_test)
    # train_knn(X_train, X_test, y_train, y_test)
    # train_svm(X_train, X_test, y_train, y_test)
    # train_mlp(X_train, X_test, y_train, y_test)
    # train_iso(X_train, X_test, y_train, y_test)
    # train_ell(X_train, X_test, y_train, y_test)
    # train_gb(X_train, X_test, y_train, y_test)
    # train_xgb(X_train, X_test, y_train, y_test)
    # train_cb(X_train, X_test, y_train, y_test)
    # train_lgb(X_train, X_test, y_train, y_test)

    train_rf_cuda(X_train, X_test, y_train, y_test)
    train_knn_cuda(X_train, X_test, y_train, y_test)
    train_svc_cuda(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
