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
from joblib import Parallel, delayed
from multiprocessing import cpu_count

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


def run_model(cur_model_name_model_pair, approach, X_train, y_train, X_test, y_test, logger=None, n_jobs=None):
    """
    Train and evaluate a single classification model.
    
    Args:
        cur_model_name_model_pair: Tuple of (model_class, model_name)
        approach: Vectorization approach name
        X_train: Training feature vectors
        y_train: Training labels
        X_test: Test feature vectors
        y_test: Test labels
        logger: Optional logger instance
        n_jobs: Number of parallel jobs for model training
    
    Returns:
        Dictionary with accuracy, f1, precision, recall metrics
    """
    def log_or_print(message, logger):
        print(message) if logger is None else logger.info(message)
        return None

    # train model
    model = cur_model_name_model_pair[0]()
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


def run_models_parallel(models, approach, X_train, y_train, X_test, y_test, logger=None, n_jobs=None):
    """
    Train and evaluate multiple models in parallel using joblib.
    
    Args:
        models: List of (model_class, model_name) tuples
        approach: Vectorization approach name
        X_train: Training feature vectors
        y_train: Training labels
        X_test: Test feature vectors
        y_test: Test labels
        logger: Optional logger instance
        n_jobs: Number of parallel jobs
    
    Returns:
        Dictionary of model results
    """
    if n_jobs is None:
        n_jobs = cpu_count()
    
    if n_jobs == 1 or len(models) < 2:
        # For single model or small number, sequential is faster
        results = {}
        for model in models:
            results[model[1]] = run_model(model, approach, X_train, y_train, X_test, y_test, logger, n_jobs=1)
        return results
    
    try:
        # Parallel execution of model training
        results = Parallel(n_jobs=n_jobs)(
            delayed(run_model)(model, approach, X_train, y_train, X_test, y_test, logger, n_jobs=1)
            for model in models
        )
        
        # Map results back to model names
        return {models[i][1]: results[i] for i in range(len(models))}
    except Exception as e:
        # Fallback to sequential if parallel execution fails
        logger.info(f'[Classifier] Parallel training failed: {e}. Using sequential processing.') if logger else print(f'[Classifier] Parallel training failed: {e}. Using sequential processing.')
        results = {}
        for model in models:
            results[model[1]] = run_model(model, approach, X_train, y_train, X_test, y_test, logger, n_jobs=1)
        return results
