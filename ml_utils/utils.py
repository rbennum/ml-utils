import pandas as pd
import numpy as np
import joblib
import os

from datetime import datetime
from typing import Optional
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def skim_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Skims the dataframe for summary statistics including nulls, negatives,
    uniques, and min/max values.

    :param DataFrame data: The input dataframe.
    :return: A dataframe contains the summary of the input dataframe.
    :rtype: DataFrame
    """
    numeric_cols = set(data.select_dtypes(include=[np.number]).columns)
    min_values = data.min()
    max_values = data.max()
    unique_counts = data.nunique()

    numeric_meta = {}
    for col in numeric_cols:
        numeric_meta[col] = {
            "neg_%": round((data[col] < 0).mean() * 100, 3),
            "zero_%": round((data[col] == 0).mean() * 100, 3),
        }

    skimmed_data = pd.DataFrame(
        {
            "feature": data.columns,
            "dtype": data.dtypes.astype(str),
            "null_%": round(data.isna().mean() * 100, 3),
            "negative_%": [
                numeric_meta.get(col, {}).get("neg_%", "-") for col in data.columns
            ],
            "zero_%": [
                numeric_meta.get(col, {}).get("zero_%", "-") for col in data.columns
            ],
            "min": [min_values.get(col, "-") for col in data.columns],
            "max": [max_values.get(col, "-") for col in data.columns],
            "n_unique": unique_counts.values,
            "unique_%": round(unique_counts / len(data) * 100, 2).values,
            "sample_values": [
                list(data[col].dropna().unique()[:5]) for col in data.columns
            ],
        }
    )

    print(f"Total duplicate rows: {data.duplicated().sum()}")
    print(f"DF shape: {data.shape}")

    return skimmed_data.reset_index(drop=True)


def save_model(model_object, filename, directory="models"):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Folder at '{directory}' has been created.")

        filepath = os.path.join(directory, filename)
        joblib.dump(model_object, filepath)
        print(f"Model object successfully saved at: {filepath}")
    except Exception as e:
        print(f"Error when trying to save the object: {e}")


def load_model(filepath):
    try:
        loaded_object = joblib.load(filepath)
        print(f"Model loaded from: {filepath}")
        return loaded_object
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
        return None
    except Exception as e:
        print(f"Error when loading object: {e}")
        return None


def get_time(time: Optional[datetime] = None) -> str:
    if time is None:
        time = datetime.now()
    timestamp_str = time.strftime("%Y_%m_%d_%H_%M_%S")
    return timestamp_str


def audit_deep(estimator, name="root", indent=0):
    """
    Recursively inspects a scikit-learn estimator, pipeline, or ColumnTransformer
    to verify the fitting status of all internal components.

    This utility traverses nested structures (like pipelines within column
    transformers) and prints a visual tree indicating which specific steps
    or transformers are currently fitted or unfitted.

    Parameters
    ----------
    estimator : estimator object
        The scikit-learn compatible estimator or meta-estimator to audit.
    name : str, default="root"
        The display name for the current level of the recursion.
    indent : int, default=0
        The current indentation level for the printed output.
    """
    space = "  " * indent
    try:
        check_is_fitted(estimator)
        print(f"{space}[✓] {name} is fitted.")
    except NotFittedError:
        print(f"{space}[X] {name} IS NOT FITTED!")

    if isinstance(estimator, Pipeline):
        for step_name, step_obj in estimator.named_steps.items():
            audit_deep(step_obj, name=f"Step: {step_name}", indent=indent + 1)

    elif isinstance(estimator, ColumnTransformer):
        if hasattr(estimator, "transformers_"):
            for trans_name, trans_obj, _ in estimator.transformers_:
                if trans_name == "remainder":
                    continue
                audit_deep(
                    trans_obj, name=f"Transformer: {trans_name}", indent=indent + 1
                )
