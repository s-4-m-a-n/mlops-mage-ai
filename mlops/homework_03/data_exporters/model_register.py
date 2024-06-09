import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
import os
import pickle

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    # Specify your data exporting logic here
    EXPERIMENT_NAME = "linear_model"
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(data)
    dv, model = data


    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(model, "linear_regression")

        dest_path = "linear_regression"
        os.makedirs(dest_path, exist_ok=True)

        # Save DictVectorizer and datasets
        dump_pickle(dv, os.path.join(dest_path, "dict_vectorizer.pkl"))
        mlflow.log_artifact(os.path.join(dest_path, "dict_vectorizer.pkl"))
