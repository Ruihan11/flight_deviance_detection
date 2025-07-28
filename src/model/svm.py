
import os
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

def train(X_train, X_test, y_train, y_test):
    """
    1. 使用 Pipeline(StandardScaler + SVC) 做网格搜索
    2. 选出最优超参后，输出 CV & 测试集指标
    3. 保存整个 Pipeline（含 scaler + 模型）
    """

    print(">> SVM")
    os.makedirs("models", exist_ok=True)

    # —— 构造 Pipeline —— 
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc",    SVC())
    ])

    # —— 超参网格 —— 
    param_grid = {
        "svc__C":      [1e-4, 1e-3, 1e-2, 1e-1, 1, 10],
        "svc__kernel": ["rbf", "poly", "sigmoid"],
        "svc__gamma":  ["scale", "auto"]
    }

    # —— 网格搜索 —— 
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )
    gs.fit(X_train, y_train)
    print("SVM best params:", gs.best_params_)
    print(f"SVM best CV accuracy: {gs.best_score_:.4f}")

    # —— 在测试集上评估 —— 
    best_pipe = gs.best_estimator_
    y_pred = best_pipe.predict(X_test)
    print(f"\nSVM Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # —— 保存 Pipeline —— 
    joblib.dump(best_pipe, "models/SVC_pipeline.pkl")
