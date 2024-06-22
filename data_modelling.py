# Author: Bethany Schoen
# Date: 9th June 2024
# -------------------- #
# CLASSIFICATION MODEL TRAINING
# -------------------- #

import pandas as pd
# splitting training and testing data
from sklearn.model_selection import train_test_split
# standardise features
from sklearn.preprocessing import StandardScaler
# model selection
from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV
# ML models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
# evaluate feature importance
import shap
# score model
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
# plotting
from matplotlib import pyplot as plt
import seaborn as sns


def create_training_and_test_data(data, feature_cols, label_col, test_size:float=0.33):
    """
    Split into train and test sets
    Standardise features
    """
    ## Train test split
    X_train, X_test, y_train, y_test = train_test_split(data[feature_cols], 
                                                        data[label_col], 
                                                        test_size=test_size, 
                                                        random_state=42, 
                                                        stratify=data[label_col]
                                                        )
    # standardize
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    return X_train, X_train_std, X_test, X_test_std, y_train, y_test, scaler

def tune_lr_model(X_train_std, y_train):

    print("## Tuning Logistic Regressor ##")
    param_grid = {
        "solver": ['newton-cg', 'liblinear'], #'lbfgs', 
        "penalty": ['l2'],
        "C": [100, 10, 1.0, 0.1, 0.01]
    }

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    random_search = RandomizedSearchCV(LogisticRegression(),
                                   param_grid, cv=cv)
    result = random_search.fit(X_train_std, y_train)

    # summarize result
    print('Best Score: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)

def train_lr_model(X_train_std, y_train, params: dict):
    """
    Fit logistic regressor with training data
    """
    # LOGISTIC REGRESSION
    model = LogisticRegression(random_state=42, 
                               solver=params["solver"],
                               penalty=params["penalty"],
                               C=params["C"]).fit(X_train_std, y_train)

    return model

def tune_rf_model(X_train_std, y_train):

    print("## Tuning Random Forest ##")
    param_grid = {
        'n_estimators': [25, 50, 100, 150],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [3, 6, 9],
        'max_leaf_nodes': [3, 6, 9],
        'min_samples_split': [2, 4, 6]
    }

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    random_search = RandomizedSearchCV(RandomForestClassifier(),
                                   param_grid, cv=cv)
    result = random_search.fit(X_train_std, y_train)

    # summarize result
    print('Best Score: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)

def train_rf_model(X_train_std, y_train, params: dict):
    """
    Fit random forest with training data
    """
    model = RandomForestClassifier(n_estimators=params["n_estimators"],
                                   max_features=params["max_features"],
                                   max_depth=params["max_depth"],
                                   max_leaf_nodes=params["max_leaf_nodes"],
                                   min_samples_split=params["min_samples_split"]).fit(X_train_std, y_train)

    return model

def tune_knn_model(X_train_std, y_train):

    print("## Tuning K-Nearest Neighbours ##")
    param_grid = {
        'n_neighbors': list(range(5, 13, 2)),
        'weights': ['uniform', 'distance'],
        'metric': ['minkowski', 'euclidean', 'manhattan']
    }

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    random_search = RandomizedSearchCV(KNeighborsClassifier(),
                                   param_grid, cv=cv)
    result = random_search.fit(X_train_std, y_train)

    # summarize result
    print('Best Score: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)

def train_knn_model(X_train_std, y_train, params: dict):
    """
    Fit KNN with training data
    """
    model = KNeighborsClassifier(n_neighbors=params["n_neighbors"],
                                 weights=params["weights"],
                                 metric=params["metric"]).fit(X_train_std, y_train)

    return model

def tune_xgb_model(X_train_std, y_train):

    print("## Tuning XGBoost Classifier ##")
    param_grid = {
        "learning_rate" : [0.05,0.10,0.15,0.20,0.25,0.30],
        "max_depth" : [ 3, 4, 5, 6, 8, 10, 12, 15],
        "min_child_weight" : [ 1, 3, 5, 7 ],
        "gamma": [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
        "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
        }

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    random_search = RandomizedSearchCV(XGBClassifier(objective='binary:logistic'),
                                   param_grid, cv=cv)
    result = random_search.fit(X_train_std, y_train)

    # summarize result
    print('Best Score: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)

def train_xgb_model(X_train_std, y_train, params: dict):
    """
    Fit xgboost classifier with training data
    """
    model = XGBClassifier(objective='binary:logistic', 
                               solver=params["learning_rate"],
                               max_depth=params["max_depth"],
                               min_child_weight=params["min_child_weight"],
                               gamma=params["gamma"],
                               colsample_bytree=params["colsample_bytree"]).fit(X_train_std, y_train)

    return model

def tune_ml_model(model_type: str, X_train_std, y_train):
    """
    Tuen all ML models and return score/params
    """
    if model_type == "lr":
        model = tune_lr_model(X_train_std, y_train)
    elif model_type == "rf":
        model = tune_rf_model(X_train_std, y_train)
    elif model_type == "knn":
        model = tune_knn_model(X_train_std, y_train)
    elif model_type == "xgb":
        model = tune_xgb_model(X_train_std, y_train)
    else:
        e = "Model argument not recognised."
        raise Exception(e)
    
    return model

def train_ml_model(model_type: str, X_train_std, y_train, params=None):
    """
    Choose which type of model to train
    """
    if model_type == "lr":     
        model = train_lr_model(X_train_std, y_train, params)
    elif model_type == "rf":
        model = train_rf_model(X_train_std, y_train, params)
    elif model_type == "knn":
        model = train_knn_model(X_train_std, y_train, params)
    elif model_type == "xgb":
        model = train_xgb_model(X_train_std, y_train, params)
    else:
        e = "Model argument not recognised."
        raise Exception(e)
    
    return model

def test_ml_model(model_type: str, model, X_test, X_test_std, y_test):
    """
    Make predictions for test set and compare to actual labels
    Print ROC curve and confusion matrix 
    Save predictions compared to actual as a csv
    """
    # Predict from the test dataset
    predictions = model.predict(X_test_std)
    predictions_probas = pd.DataFrame(model.predict_proba(X_test_std), columns=["proba0", "proba1"])
    predictions_probas["chosen_class_proba"] = predictions_probas.apply(lambda row: max(row["proba0"], row["proba1"]), axis=1)

    # Precision Recall scores
    print("Precision and Recall scores in testing\n")
    print(classification_report(y_test, predictions, digits=3))

    # roc score
    print("ROC Score\n")
    print(roc_auc_score(y_test, predictions))
    print("F1 Score:")
    f1_result = f1_score(y_test, predictions)
    print(f1_result)

    # Confusion matrix
    print("\nConfusion Matrix")
    fig, ax = plt.subplots()
    cf_matrix = confusion_matrix(y_test, predictions)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=[0, 1])

    cm_display.plot(ax=ax)
    ax.set_title(f"{model_type} Confusion Matrix")
    plt.show()

    return f1_result

def evaluate_shap(model_type, X_test_std, model, feature_cols):
    """
    Study the importance of each features when making the predictions
    """
    X_test_with_columns_headers = pd.DataFrame(X_test_std, columns=feature_cols)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_with_columns_headers)
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test_with_columns_headers, plot_type='bar', plot_size=None)

def tune_models_main(data, feature_cols, label_col):
    X_train, X_train_std, X_test, X_test_std, y_train, y_test, scaler = create_training_and_test_data(data, feature_cols, label_col)
    for model_type in ["knn", "lr", "xgb"]:
        i = 0
        while i < 3:
            print("Attempt ", i)
            tune_ml_model(model_type, X_train_std, y_train)    
            i += 1
            print("\n")

    print("\n")

    return

def compare_models_main(data, feature_cols, label_col, params_dict):
    X_train, X_train_std, X_test, X_test_std, y_train, y_test, scaler = create_training_and_test_data(data, feature_cols, label_col)
    results = []
    for model_type in ["knn", "lr", "xgb"]:
        print("## {} ##".format(model_type))
        params = params_dict[model_type]
        model = train_ml_model(model_type, X_train_std, y_train, params)
        f1_result = test_ml_model(model_type, model, X_test, X_test_std, y_test)
        results.append((model_type, f1_result))

    print(results)

    return

def modelling_main(data, feature_cols, label_col, model_type, params):
    X_train, X_train_std, X_test, X_test_std, y_train, y_test, scaler = create_training_and_test_data(data, feature_cols, label_col)
    model = train_ml_model(model_type, X_train_std, y_train, params)
    if model_type in ["xgb", "rf"]:
        evaluate_shap(model_type, X_test_std, model, feature_cols)

    return model

if __name__ == "__main__":
    compare_models_main()