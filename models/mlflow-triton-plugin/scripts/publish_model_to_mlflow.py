import mlflow
import onnx
import mlflow.onnx
import os 
from random import randint
from random import random
import click

import tensorrt_flavor, fil_flavor


@click.command()
@click.option(
    "--model_name",
    help="Model name",
)
@click.option(
    "--model_file",
    type=click.Path(exists=True, readable=True),
    required=True,
    help="Model filepath"
)
@click.option(
    "--model_config",
    type=click.Path(exists=True, readable=True),
    required=True,
    help="Model config filepath"
)
@click.option(
    "--flavor",
    type=click.Choice(['onnx', 'tensorrt', 'fil'], case_sensitive=True),
    required=True,
    help="Model flavor",
)
def publish_to_mlflow(
    model_name,
    model_file,
    model_config,
    flavor
):
    mlflow_tracking_uri=os.environ['MLFLOW_TRACKING_URI']
    artifact_path = "triton"

    mlflow.set_tracking_uri(uri=mlflow_tracking_uri)

    with mlflow.start_run() as run: 
        
        mlflow.log_artifact(model_config, artifact_path=artifact_path)
        if flavor=="onnx":
            model = onnx.load(model_file)
            mlflow.onnx.log_model(model,
                                    artifact_path=artifact_path,
                                    registered_model_name=model_name,
                                    )
        elif flavor=="tensorrt":
            tensorrt_flavor.log_model(model_file,
                                artifact_path=artifact_path,
                                registered_model_name=model_name,
                                )
        elif flavor=="fil":
            fil_flavor.log_model(model_file,
                                artifact_path=artifact_path,
                                registered_model_name=model_name,
                                )

        print(mlflow.get_artifact_uri())

if __name__ == "__main__":
    publish_to_mlflow()