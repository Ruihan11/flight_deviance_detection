
import os
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train(X_train, X_test, y_train, y_test):
    """
    1. 5‑折交叉验证选最佳 k
    2. 用最佳 k 在全量训练集上训练
    3. 在测试集上评估并保存模型
    """
    print(">> KNN")
    os.makedirs("models", exist_ok=True)

    # 如果 X_train 还是 DataFrame，转成 numpy 数组
    X = X_train.values if hasattr(X_train, "values") else X_train
    y = y_train.values if hasattr(y_train, "values") else y_train

    param_list = list(range(2, 31, 2))
    best_k, best_score = None, 0.0

    # —— 交叉验证寻找最佳 k —— 
    for k in param_list:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in cv.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            clf = KNeighborsClassifier(n_neighbors=k)
            clf.fit(X_tr, y_tr)
            scores.append(clf.score(X_val, y_val))

        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_k = k

    print(f"KNN best k = {best_k}, CV accuracy = {best_score:.4f}")

    # —— 用最佳 k 训练最终模型 —— 
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X, y)

    # —— 测试集评估 —— 
    X_t = X_test.values if hasattr(X_test, "values") else X_test
    y_pred = knn.predict(X_t)
    print(f"\nKNN Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:\n", classification_report(
        y_test, y_pred,
        digits=4,
        zero_division=0
    ))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # —— 保存模型 —— 
    joblib.dump(knn, "models/KNearestNeighbor.pkl")
