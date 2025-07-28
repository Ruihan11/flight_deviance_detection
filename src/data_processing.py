import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_split_data(
    planes_csv: str,
    labels_csv: str,
    test_size: float = 0.2,
    random_state: int = 42
):
    # 1. 读表
    planes = pd.read_csv(planes_csv)
    labels = pd.read_csv(labels_csv)

    # 2. 预处理 Tail → surveil
    labels['registration'] = labels['registration'].astype(str).str.upper().str.strip()
    unique_tails = (
        planes['Tail'].dropna()
        .astype(str).str.upper().str.strip()
        .unique()
    )
    n = len(unique_tails)
    surveil_vals = np.array([1]*(n//10) + [0]*(n - n//10))
    np.random.shuffle(surveil_vals)
    mapping = dict(zip(unique_tails, surveil_vals))

    planes['Tail']    = planes['Tail'].astype(str).str.upper().str.strip()
    planes['surveil'] = planes['Tail'].map(mapping).fillna(0).astype(int)

    # 3. 特征与标签
    features = ['mean_speed','mean_changeofheading','duration','distance','type_code']
    X = planes[features]
    y = planes['surveil']

    # 4. 划分
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
