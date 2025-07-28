from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

def train(X_train, X_test, y_train, y_test):
    """
    使用随机森林模型（RandomForestClassifier）对输入数据进行训练、调参和评估。
    1. OOB曲线：探索不同 n_estimators 设置下模型性能；
    2. 网格搜索：使用 GridSearchCV 寻找最佳超参数组合；
    3. 模型训练与评估：输出准确率、分类报告与混淆矩阵；
    4. 模型持久化：将最终模型保存为 .pkl 文件。
    """
    print(">> RANDOM FOREST")
    os.makedirs("models", exist_ok=True)

    # —— 1. OOB 曲线示例（可选） —— 
    oob_scores = []
    for n in tqdm(range(20, 1301, 20), desc="RF: training n_estimators"):
        m = RandomForestClassifier(
            n_estimators=n, oob_score=True,
            random_state=42, n_jobs=-1
        )
        m.fit(X_train, y_train)
        oob_scores.append(m.oob_score_)
    # （可在此处保存 oob_scores 或画图）

    # —— 2. 网格搜索 —— 
    param_grid = {
        'n_estimators': [400,500,600,700,800],
        'max_depth':    [10,20,None],
        'min_samples_leaf':[1,5],
        'max_features':['sqrt','log2']
    }
    gs = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid, cv=3, scoring='accuracy', n_jobs=-1
    )
    gs.fit(X_train, y_train)
    print("RF best params:", gs.best_params_, "best score:", gs.best_score_)

    # —— 3. 最优模型 —— 
    best = gs.best_estimator_
    best.fit(X_train, y_train)
    pred = best.predict(X_test)
    print("RF Test acc:", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred, digits=4))
    print(confusion_matrix(y_test, pred))

    # —— 4. 保存模型 —— 
    joblib.dump(best, "models/RandomForestClassifier.pkl")
