import pandas as pd
import numpy as np
from typing import List
from prefect import flow, get_run_logger
from prefect_dask import DaskTaskRunner
from sklearn.model_selection import train_test_split
from typing import Tuple


from tasks.print_evaluation import print_evaluation
from tasks.train_model import train_model, train_and_evaluate_alpha
from tasks.standartscaler import scale_features


@flow(flow_run_name="{method}-{scale}-{alpha}-{file_name}-{drop_columns}-{pop}")
async def train_model_flow(sample: int = 0, 
                           method: str = "Ridge", 
                           alpha: float = 0, 
                           scale: bool = False, 
                           file_name: str = "immo_data_preprocessed.csv", 
                           drop_columns: List[str] = [""],
                           pop: str = "baseRent"):

    
    df = pd.read_csv(f"./Daten/{file_name}")

    if not drop_columns == [""]:
        df = df.drop(columns=drop_columns)
    
    if sample > 0:
        df = df.sample(10000, random_state=42)

    y = df.pop(f"{pop}")
    df = pd.get_dummies(df)
    y = y.dropna()
    
    X_train, X_test, y_train, y_test = train_test_split(df, 
                                                        y, 
                                                        test_size=0.2, 
                                                        random_state=42)

    if scale == True:
        X_train, X_test = await scale_features(X_train, X_test)

    y_train_pred, y_test_pred, model = await train_model(X_train, 
                                                         X_test, 
                                                         y_train, 
                                                         method=method, 
                                                         alpha=alpha)
    
    await print_evaluation(model, 
                           X_train, 
                           X_test, 
                           y_train, 
                           y_test, 
                           y_train_pred, 
                           y_test_pred, 
                           alpha,
                           feature_names=df.columns
                           )


@flow(flow_run_name="{method}-{scale}-{interval}-{file_name}-{drop_columns}-{pop}", task_runner=DaskTaskRunner())
async def grid_search(interval: Tuple[float, float, int], 
                      method: str = "Ridge", 
                      scale: bool = False, 
                      sample: int = 0, 
                      file_name: str = "immo_data_preprocessed.csv", 
                      drop_columns: List[str] = [""],
                      pop: str = "baseRent" ):
    
    logger = get_run_logger()
    
    df = pd.read_csv(f"./Daten/{file_name}")

    if not drop_columns == [""]:
        df = df.drop(columns=drop_columns)

    if sample > 0:
        df = df.sample(10000, random_state=42)

    df = pd.get_dummies(df)
    y = df.pop(f"{pop}")

    X_train, X_test, y_train, y_test = train_test_split(df, 
                                                        y, 
                                                        test_size=0.2, 
                                                        random_state=42)

    if scale:
        X_train, X_test = await scale_features(X_train, X_test)

    min_alpha, max_alpha, num_alphas = interval
    alpha_values = np.linspace(min_alpha, max_alpha, num_alphas)

    futures = train_and_evaluate_alpha.map(
        alpha = alpha_values, 
        X_train=[X_train]*len(alpha_values),
        X_test=[X_test]*len(alpha_values),
        y_train=[y_train]*len(alpha_values),
        y_test=[y_test]*len(alpha_values),
        method=[method]*len(alpha_values),
        feature_names=[df.columns]*len(alpha_values),
        logger=logger
    )

    results = futures.result()
    results_df = pd.DataFrame(results)

    return results_df