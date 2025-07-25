import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer, mean_absolute_error



def load_datasets(train_path, test_path, sampl6_path, novartis_path):
    """
    loads the training (training_split.csv)
    and test data (test_split.csv, sampl6.csv, novartis.csv)
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sampl6 = pd.read_csv(sampl6_path)
    novartis = pd.read_csv(novartis_path)
    
    return train, test, sampl6, novartis


def preprocess_data(data):
    """
    get features (x) and exp_pKa (y)
    """
    if 'predicted_rf (pKa)' in data.columns:
        data = data.drop('predicted_rf (pKa)', axis=1)
    
    y = data['exp_pKa (pKa)']    
    x = data.drop(['file', 'conjugate_acid', 'conjugate_base', 'exp_pKa (pKa)'], axis=1)

    return x, y


def train_rf(train_x, train_y):
    """
    train random forest model
    """
    rf = RandomForestRegressor(n_estimators=2000, max_features=0.5, max_depth=40, min_samples_split=2, min_samples_leaf=2, random_state=42)
    trained_model = rf.fit(train_x, train_y)
    
    return trained_model


def evaluate_rf(trained_model, test_x, test_y):
    """
    get mean absolute error (MAE),
    get root mean squared error (RMSE),
    get coefficient of determination (R^2)
    """
    pred = trained_model.predict(test_x)
    mae = mean_absolute_error(test_y, pred)
    rmse = mean_squared_error(test_y, pred, squared=False)
    r2 = r2_score(test_y, pred)    
    
    return mae, rmse, r2


def plot_feature_importance(model, feature_names, top_n=20):
    """
    Plots the top N feature importances
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 6))
    plt.title(f"Top {top_n} Feature Importances")
    plt.bar(range(top_n), importances[indices], align="center")
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()

    
def run_model():
    """
    loads the datasets, trains the RF model, and predicts the pKas for
    test split, sampl6, and novartis
    """ 
    
    # load the datasets
    train, test, sampl6, novartis = load_datasets('../datasets/train_split.csv', '../datasets/test_split.csv', '../datasets/sampl6.csv', '../datasets/novartis.csv')   
 
    # get features and exp_pKa of datasets
    train_x, train_y = preprocess_data(train)
    test_x, test_y = preprocess_data(test)
    sampl6_x, sampl6_y = preprocess_data(sampl6)
    novartis_x, novartis_y = preprocess_data(novartis)
    
    # train random forest model
    trained_model = train_rf(train_x, train_y)
    
    # 🔥 Plot feature importances
    plot_feature_importance(trained_model, train_x.columns)
    
    # evaluate trained model on test sets
    test_mae, test_rmse, test_r2 = evaluate_rf(trained_model, test_x, test_y)
    sampl6_mae, sampl6_rmse, sampl6_r2 = evaluate_rf(trained_model, sampl6_x, sampl6_y)
    novartis_mae, novartis_rmse, novartis_r2 = evaluate_rf(trained_model, novartis_x, novartis_y)

    test_set_results = [
        f'test MAE: {round(test_mae, 2)}',
        f'test RMSE: {round(test_rmse, 2)}',
        f'test R^2: {round(test_r2, 2)}',
        f'sampl6 MAE: {round(sampl6_mae, 2)}',
        f'sampl6 RMSE: {round(sampl6_rmse, 2)}',
        f'sampl6 R^2: {round(sampl6_r2, 2)}',
        f'novartis MAE: {round(novartis_mae, 2)}',
        f'novartis RMSE: {round(novartis_rmse, 2)}',
        f'novartis R^2: {round(novartis_r2, 2)}',

    ]

    
    return test_set_results


print(run_model())

