from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

VECTORIZATION_APPROACHES = [
    'review_vector__simple',
    'review_vector__tfidf',
    'review_vector__word2vec',
    'review_vector__glove',
]

CLASSIFICATION_MODELS = [
    [LogisticRegression, "Logistic Regression"],
    [RandomForestClassifier, "Random Forest"],
    [SVC, "SVM"],
    [GradientBoostingClassifier, "Gradient Boosting"],
    [KNeighborsClassifier, "KNN"],
    [DecisionTreeClassifier, "Decision Tree"],
    [MLPClassifier, "MLP"],
    [GaussianNB, "Naive Bayes"],
    [AdaBoostClassifier, "AdaBoost"],
    [XGBClassifier, "XGBoost"],
    [LGBMClassifier, "LightGBM"],
    [CatBoostClassifier, "CatBoost"],
]


def run_model(cur_model_name_model_pair, approach, X_train, y_train, X_test, y_test, logger=None):

    def log_or_print(message, logger):
        print(message) if logger is None else logger.info(message)
        return None

    # train model
    model = cur_model_name_model_pair[0]()
    # log_or_print(f'[Train] [approach: {approach}] [Model: {cur_model[1]}] Start fitting', logger)
    log_or_print(
        f'[Train] [approach: {approach}] [Model: {cur_model_name_model_pair[1]}] Start fitting', logger)
    model.fit(X_train.tolist(), y_train)
    log_or_print(
        f'[Train] [approach: {approach}] [Model: {cur_model_name_model_pair[1]}] Fitting completed', logger)

    # predict
    y_pred = model.predict(X_test.tolist())

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    # evaluate by acc, f1, precision, recall
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
