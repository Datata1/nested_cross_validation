import pandas as pd
import numpy as np
from typing import List
from prefect import flow, get_run_logger
from prefect_dask import DaskTaskRunner
from sklearn.model_selection import train_test_split
from typing import Tuple


from flows.aufgabe1 import train_model_flow
from tasks.train_model import train_and_evaluate_alpha
from tasks.standartscaler import scale_features

from sklearn.model_selection import KFold


@flow(task_runner=DaskTaskRunner())
async def cross_validation(file_name: str = "immo_data_preprocessed.csv",
                           drop_columns: List[str] = [],
                           interval: Tuple[float, float, int] = (0.01, 10, 100),
                           method: str = "Ridge",
                           scale: bool = False,
                           pop: str = "baseRent",
                           sample: int = 0,
                           n_splits: int = 3
                           ):
    
    logger = get_run_logger()
    
    df = pd.read_csv(f"./Daten/{file_name}")

    if not drop_columns == [""]:
        df = df.drop(columns=drop_columns)

    if sample > 0:
        df = df.sample(10000, random_state=42)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    df = pd.get_dummies(df)
    y = df.pop(f"{pop}")

    performance_dict = {alpha: [] for alpha in np.linspace(interval[0], interval[1], interval[2])}

    for fold, (train_indices, validation_indices) in enumerate(kf.split(df)):
        logger.info(f"Fold {fold + 1}:")
        logger.info("-------")
        logger.info(f"Indizes der Trainingsdaten: {train_indices}.")
        logger.info(f"Indizes der Validierungsdaten: {validation_indices}")

        X_train = df.iloc[train_indices]
        X_validation = df.iloc[validation_indices]
        
        y_train = y.iloc[train_indices]
        y_validation = y.iloc[validation_indices]

        if scale:
            X_train, X_validation = await scale_features(X_train, X_validation)

        min_alpha, max_alpha, num_alphas = interval
        alpha_values = np.linspace(min_alpha, max_alpha, num_alphas)

        futures = train_and_evaluate_alpha.map(
            alpha=alpha_values, 
            X_train=[X_train]*len(alpha_values),
            X_test=[X_validation]*len(alpha_values),
            y_train=[y_train]*len(alpha_values),
            y_test=[y_validation]*len(alpha_values),
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
    logger.info(f"Beste Alpha: {best_alpha} mit durchschnittlichem RMSE: {average_performance[best_alpha]}")

    await train_model_flow(file_name=file_name, 
                            drop_columns=drop_columns,
                            method=method,
                            scale=scale,
                            pop=pop,
                            sample=sample,
                            alpha=best_alpha)
    
    return average_performance, best_alpha
 