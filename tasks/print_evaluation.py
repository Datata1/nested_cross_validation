import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn import linear_model
from prefect import task, get_run_logger
from prefect.artifacts import create_markdown_artifact

@task()
async def print_evaluation(model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, alpha, feature_names):
    """ Ausgabe von R2-Wert, MSE und MAE für Trainings- und Testset """
    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = root_mean_squared_error(y_train, y_train_pred) 
    mae_train = mean_absolute_error(y_train, y_train_pred)

    r2_test = r2_score(y_test, y_test_pred)
    rmse_test = root_mean_squared_error(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    
    # Log output
    logger = get_run_logger()
    logger.info(
        f"{model} Evaluation:\n"
        f"{'':6} {'R²':>10} | {'RMSE':>14} | {'MAE':>10} | {'rows':>8} | {'columns':>8}\n"
        f"{'Train':6} {r2_train:10.5f} | {rmse_train:14.2f} | {mae_train:10.2f} | {X_train.shape[0]:8} | {X_train.shape[1]:8}\n"
        f"{'Test':6} {r2_test:10.5f} | {rmse_test:14.2f} | {mae_test:10.2f} | {X_test.shape[0]:8} | {X_test.shape[1]:8}\n"
    )
    
    # Create markdown report
    markdown_content = (
        f"### {model} Evaluation\n\n"
        f"| Dataset | R² | RMSE | MAE | Rows | Columns |\n"
        f"|---------|--------:|------------:|--------:|-------:|-------:|\n"
        f"| Train   | {r2_train:.5f} | {rmse_train:.2f} | {mae_train:.2f} | {X_train.shape[0]} | {X_train.shape[1]} |\n"
        f"| Test    | {r2_test:.5f} | {rmse_test:.2f} | {mae_test:.2f} | {X_test.shape[0]} | {X_test.shape[1]} |\n\n"
    )

    # Sort and display the top 10 coefficients
    coefficients_lr = pd.DataFrame({"Feature Name": feature_names, "Coefficient": model.coef_})
    top_features = coefficients_lr.reindex(coefficients_lr["Coefficient"].abs().sort_values(ascending=False).index).head(10)
    
    # Append top 10 coefficients to markdown
    markdown_content += "### Top 10 Coefficients\n\n"
    markdown_content += top_features.to_markdown(index=False)
    
    # Append count of zero coefficients
    zero_count = len(coefficients_lr[np.isclose(coefficients_lr.Coefficient, 0.0)])
    markdown_content += f"\n\nNumber of coefficients that are zero: {zero_count}/{len(coefficients_lr)}\n"

    # Append alpha value
    markdown_content += f"\n\nAlpha value: {alpha}\n"
    
    # Create markdown artifact in Prefect
    await create_markdown_artifact(
        key="evaluation",
        markdown=markdown_content,
        description="Model Evaluation"
    )
