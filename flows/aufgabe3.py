from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from prefect import flow
from typing import List, Tuple
from prefect_dask import DaskTaskRunner
from prefect import get_run_logger
from sklearn.metrics import root_mean_squared_error


from flows.aufgabe1 import train_model_flow
from tasks.train_model import train_and_evaluate_alpha, train_model
from tasks.standartscaler import scale_features

@flow(flow_run_name="{method}-{scale}-{interval}-{file_name}-{drop_columns}-{pop}", task_runner=DaskTaskRunner())
async def nested_cross_validation(file_name: str = "immo_data_preprocessed.csv",
                                   drop_columns: List[str] = [],
                                   interval: Tuple[float, float, int] = (1, 100, 50),
                                   method: str = "Ridge",
                                   scale: bool = False,
                                   pop: str = "baseRent",
                                   sample: int = 0,
                                   outer_n_splits: int = 5,
                                   inner_n_splits: int = 3):
    
    logger = get_run_logger()

    # Lade die Daten
    df = pd.read_csv(f"./Daten/{file_name}")

    if not drop_columns == [""]:
        df = df.drop(columns=drop_columns)

    if sample > 0:
        df = df.sample(10000, random_state=42)

    df = pd.get_dummies(df)
    y = df.pop(pop)

    outer_kf = KFold(n_splits=outer_n_splits, shuffle=True, random_state=42)
    
    test_errors = []  
    best_alphas = []  

    for outer_fold, (train_indices, test_indices) in enumerate(outer_kf.split(df)):
        logger.info(f"Outer Fold {outer_fold + 1}:")
        logger.info("-------")
        
        X_train = df.iloc[train_indices]
        X_test = df.iloc[test_indices]
        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]

        inner_kf = KFold(n_splits=inner_n_splits, shuffle=True, random_state=42)
        min_alpha, max_alpha, num_alphas = interval
        performance_dict = {alpha: [] for alpha in np.linspace(min_alpha, max_alpha, num_alphas)}

        for inner_fold, (inner_train_indices, inner_val_indices) in enumerate(inner_kf.split(X_train)):
            logger.info(f"  Inner Fold {inner_fold + 1}:")
            logger.info("  -------")

            X_inner_train = X_train.iloc[inner_train_indices]
            X_inner_val = X_train.iloc[inner_val_indices]
            y_inner_train = y_train.iloc[inner_train_indices]
            y_inner_val = y_train.iloc[inner_val_indices]

            if scale:
                X_inner_train, X_inner_val = await scale_features(X_inner_train, X_inner_val)

            min_alpha, max_alpha, num_alphas = interval
            alpha_values = np.linspace(min_alpha, max_alpha, num_alphas)

            futures = train_and_evaluate_alpha.map(
                alpha=alpha_values, 
                X_train=[X_inner_train]*len(alpha_values),
                X_test=[X_inner_val]*len(alpha_values),
                y_train=[y_inner_train]*len(alpha_values),
                y_test=[y_inner_val]*len(alpha_values),
                method=[method]*len(alpha_values),
                feature_names=[df.columns]*len(alpha_values),
                logger=logger
            )

            results = futures.result()
            results_df = pd.DataFrame(results)

            for _, row in results_df.iterrows():
                performance_dict[row['Alpha']].append(row['RMSE Test'])

        average_performance = {alpha: np.mean(perf) for alpha, perf in performance_dict.items()}
        best_alpha = min(average_performance, key=average_performance.get)  # Minimiert RMSE
        best_alphas.append(best_alpha)  # Speichere die beste Alpha
        logger.info(f"  Beste Alpha: {best_alpha} mit durchschnittlichem RMSE: {average_performance[best_alpha]}")

        _, y_test_pred, _ = await train_model(X_train, X_test, y_train, method=method, alpha=best_alpha)

        logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        
        test_error = root_mean_squared_error(y_test, y_test_pred)
        test_errors.append(test_error)
        logger.info(f"  Testfehler für Outer Fold {outer_fold + 1}: {test_error}")

    average_test_error = np.mean(test_errors)
    logger.info(f"Durchschnittlicher Testfehler über alle Outer Folds: {average_test_error}")

    return {
        "average_test_error": average_test_error,
        "test_errors": test_errors,
        "best_alphas": best_alphas
    }

