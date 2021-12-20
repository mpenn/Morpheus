from setuptools import setup, find_packages


setup(
    name="mlflow-triton",
    version="0.1.0",
    description="Triton Mlflow Deployment",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "mlflow>=1.15.0",
    ],
    entry_points={"mlflow.deployments": "triton=mlflow_triton.deployments"},
)