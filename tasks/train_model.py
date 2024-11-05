from prefect import task, get_run_logger
from prefect.artifacts import create_markdown_artifact
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


from tasks.print_evaluation import print_evaluation


@task()
async def train_model(X_train, X_test, y_train, method, alpha):

    match method:
        case "Linear Regression":
            model = linear_model.LinearRegression()
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

        case "Lasso":
            model = linear_model.Lasso(alpha=alpha)
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

        case "Ridge":
            model = linear_model.Ridge(alpha=alpha)
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

    return y_train_pred, y_test_pred, model


@task
async def train_and_evaluate_alpha(alpha, X_train, X_test, y_train, y_test, method, feature_names, logger):
    y_train_pred, y_test_pred, model = await train_model(X_train, X_test, y_train, method=method, alpha=alpha)
    
    await print_evaluation(model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, alpha, feature_names=feature_names)

    result = {
        "Alpha": alpha,
        "Model": method,
        "RMSE Train": root_mean_squared_error(y_train, y_train_pred),
        "RMSE Test": root_mean_squared_error(y_test, y_test_pred),
        "R2 Train": r2_score(y_train, y_train_pred),
        "R2 Test": r2_score(y_test, y_test_pred),
        "MAE Train": mean_absolute_error(y_train, y_train_pred),
        "MAE Test": mean_absolute_error(y_test, y_test_pred)
    }

    logger.info(f"result: {result}")
    
    
    return result