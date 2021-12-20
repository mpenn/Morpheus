import os
import shutil
import logging
import requests
from pathlib import Path

from mlflow_triton.config import Config

from mlflow.deployments import BaseDeploymentClient, get_deploy_client
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.models.model import Model

import glob
import json

logger = logging.getLogger(__name__)

_MLFLOW_META_FILENAME = "mlflow-meta.json"

class TritonPlugin(BaseDeploymentClient):
    def __init__(self, uri):
        """
        Initializes the deployment plugin, sets the triton model repo
        """
        super(TritonPlugin, self).__init__(target_uri=uri)
        self.server_config = Config()
        self.triton_url, self.triton_model_repo = self._get_triton_server_config()
        self.supported_flavors = ['onnx', 'tensorrt', 'fil'] # need to add other flavors

    def _get_triton_server_config(self):
        triton_url = "http://localhost:8000"
        if self.server_config["triton_url"]:
            triton_url = self.server_config["triton_url"]
        logger.info("Triton url = {}".format(triton_url))

        if not self.server_config["triton_model_repo"]:
            raise Exception("Check that environment variable TRITON_MODEL_REPO is set")
        triton_model_repo = self.server_config["triton_model_repo"]
        logger.info("Triton model repo = {}".format(triton_model_repo))

        return triton_url, triton_model_repo

    def create_deployment(self, name, model_uri, flavor=None, config=None):
        """
        Deploy the model at the model_uri to the Triton model repo. Associated config.pbtxt and *labels* files will be deployed.

        :param name: Name of the of the model
        :param model_uri: Model uri in format model:/<model-name>/<version-or-stage>
        :param flavor: Flavor of the deployed model
        :param config: Configuration parameters

        :return: Model flavor and name
        """
        # Set the version
        self._validate_config_args(config)
        version = config['version']

        self._validate_flavor(flavor)

        # Verify model does not already exist in Triton
        if self._model_exists(name):
            raise Exception(
                "Unable to create deployment for name %s because it already exists."
                % (name)
            )
        
        # Get the path of the artifact
        path = Path(_download_artifact_from_uri(model_uri))

        copy_paths = self._get_copy_paths(path, version, name)

        self._copy_files_to_triton_repo(path, version, name)

        self._generate_mlflow_meta_file(version, name, flavor, model_uri)

        resp = requests.post(f'{self.triton_url}/v2/repository/models/{name}/load')
        if resp.status_code != 200:
            raise Exception(
                "Unable to create deployment for name %s. "
                "Server returned status code %s and response: %s"
                % (name, resp.status_code, resp.content)
            )

        return {"name": name, "flavor": flavor}
    
    
    def delete_deployment(self, name):
        """
        Delete the deployed model in Triton with the provided model name

        :param name: Name of the of the model with version number. For ex: "densenet_onnx/2"

        :return: None
        """
        model_name, version = self._validate_model_name(name)

        # Verify model is already deployed to Triton
        if not self._model_exists(model_name):
            raise Exception(
                "Unable to delete deployment for name %s because it does not exist."
                % (name)
            )

        resp = requests.post(f'{self.triton_url}/v2/repository/models/{model_name}/unload')
        if resp.status_code != 200:
            raise Exception(
                "Unable to delete deployment for name %s. "
                "Server returned status code %s and response: %s"
                % (name, resp.status_code, resp.content)
            )

        self._delete_deployment_files(name)
        
        return None

    def update_deployment(self, name, model_uri=None, flavor=None, config=None):
        """
        Update the model deployment in triton with the provided name

        :param name: Name and version number of the model, <model_name>/<version>.
        :param model_uri: Model uri models:/model_name/version
        :param flavor: The flavor of the model
        :param config: Configuration parameters

        :return: Returns the flavor of the model
        """
        # TODO: Update this function with a warning. If config and label files associated with this
        # updated model are different than the ones already deployed to triton, issue a warning to the user.

        # Set the version
        model_name, version = self._validate_model_name(name)

        self._validate_flavor(flavor)

        # Verify model is already deployed to Triton
        if not self._model_exists(model_name):
            raise Exception(
                "Unable to update deployment for name %s because it does not exist."
                % (name)
            )

        prev = self.get_deployment(model_name)

        # Get the path of the artifact
        path = Path(_download_artifact_from_uri(model_uri))

        self._copy_files_to_triton_repo(path, version, model_name)

        self._generate_mlflow_meta_file(version, model_name, flavor, model_uri)

        resp = requests.post(f'{self.triton_url}/v2/repository/models/{model_name}/load')
        if resp.status_code != 200:
            self._delete_deployment_files(name)
            # restore previous
            self.create_deployment(prev['name'], prev['mlflow_model_uri'], flavor=prev['flavor'], config={'version': prev['version']})
            raise Exception(
                "Unable to update deployment for name %s. "
                "Server returned status code %s and response: %s"
                % (name, resp.status_code, resp.content)
            )

        return {"flavor": flavor}

    def list_deployments(self):
        """
        List models deployed to Triton.

        :return: None
        """
        resp = requests.post(f'{self.triton_url}/v2/repository/index').json()
        actives = []
        for d in resp:
            if 'state' in d and d['state']=='READY':
                mlflow_meta_path = os.path.join(self.triton_model_repo, d['name'], _MLFLOW_META_FILENAME)
                if os.path.isfile(mlflow_meta_path):
                    meta_dict = self._get_mlflow_meta_dict(d['name'])
                    d['triton_model_path'] = meta_dict['triton_model_path']
                    d['mlflow_model_uri'] = meta_dict['mlflow_model_uri']
                    d['flavor'] = meta_dict['flavor']
                    actives.append(d)

        return actives

    def get_deployment(self, name):
        """
        Get deployment from Triton.

        :param name: Name of the model. \n
                     Ex: "mini_bert_onnx" - gets the details of active version of this model \n

        :return: output - Returns a dict with model info
        """
        deployments = self.list_deployments()
        for d in deployments:
            if d['name'] == name:
                return d
        raise ValueError(
            f'Unable to get deployment with name {name}'
        )

    def predict(self, deployment_name, df):
        raise NotImplementedError("predict has not been implemented yet")
        return None

    def _generate_mlflow_meta_file(self, version, name, flavor, model_uri):
        triton_deployment_dir = os.path.join(self.triton_model_repo, name)
        triton_model_path = os.path.join(triton_deployment_dir,version)
        meta_dict = {
            'name' : name,
            'version' : version,
            'triton_model_path': triton_model_path,
            'mlflow_model_uri' : model_uri,
            'flavor' : flavor
        }
        with open(os.path.join(triton_deployment_dir, _MLFLOW_META_FILENAME), "w") as outfile:
            json.dump(meta_dict, outfile, indent=4)

        print("Saved", _MLFLOW_META_FILENAME, "to", triton_deployment_dir)

    def _get_mlflow_meta_dict(self, name):
        mlflow_meta_path = os.path.join(self.triton_model_repo, name, _MLFLOW_META_FILENAME)
        with open(mlflow_meta_path, 'r') as metafile:
            mlflow_meta_dict = json.load(metafile)

        return mlflow_meta_dict

    def _get_copy_paths(self, artifact_path, version, name):
        copy_paths = {}
        copy_paths['model_path'] = {}
        copy_paths['config_path'] = {}
        for file in artifact_path.iterdir():
            if file.name not in ['requirements.txt', 'config.pbtxt', 'MLmodel', 'conda.yaml']:
                copy_paths['model_path']['from'] = file
            if file.name == 'config.pbtxt':
                copy_paths['config_path']['from'] = file
        
        triton_deployment_dir = "{}/{}".format(self.triton_model_repo,name)
        copy_paths['config_path']['to'] = triton_deployment_dir
        copy_paths['model_path']['to'] = "{}/{}".format(triton_deployment_dir,version)
        return copy_paths

    def _copy_files_to_triton_repo(self, artifact_path, version, name):
        copy_paths = self._get_copy_paths(artifact_path, version, name)
        for key in copy_paths:
            if not os.path.isdir(copy_paths[key]['to']):
                os.makedirs(copy_paths[key]['to'])
            shutil.copy(copy_paths[key]['from'], copy_paths[key]['to'])
            print("Copied", copy_paths[key]['from'], "to", copy_paths[key]['to'])
        return copy_paths

    def _delete_deployment_files(self, name):
        model_name, version = self._validate_model_name(name)
        triton_deployment_dir = os.path.join(self.triton_model_repo, name)

        # Check if the deployment directory exists
        if not os.path.isdir(triton_deployment_dir):
            raise Exception("A deployment does not exist for this model in directory {} for model name {}".format(triton_deployment_dir, name))

        model_file = glob.glob("{}/model*".format(triton_deployment_dir))
        for file in model_file:
            if os.path.isfile(file):
                print("Model file found: {}".format(file))
                os.remove(file)
                print("Model file removed: {}".format(file))

       # Delete mlflow meta file
        mlflow_meta_path = os.path.join(self.triton_model_repo, model_name, _MLFLOW_META_FILENAME)
        if os.path.isfile(mlflow_meta_path):
            os.remove(mlflow_meta_path)
    
    def _validate_config_args(self, config):
        if not config['version']:
            raise Exception("Please provide the version as a config argument")
        if not config['version'].isdigit():
            raise ValueError("Please make sure version is a number. version = {}".format(config['version']))
    
    def _validate_flavor(self, flavor):
        if flavor not in self.supported_flavors:
            raise Exception("{} model flavor not supported by Triton".format(flavor))

    def _validate_model_name(self,name):
        """Expecting model name in format model/1"""
        model = ""
        version = ""
        if "/" in name:
            split_name = name.split("/")
            model = split_name[0]
            version = split_name[1]
            if not version.isdigit():
                raise Exception("Improper format used for model name. Please use format <model_name>/<version>")
        else:
            raise Exception("Improper format used for model name. Please use format <model_name>/<version>")
        return model, version

    def _model_exists(self, name):
        deploys = self.list_deployments()
        exists = False
        for d in deploys:
            if d['name']==name:
                exists = True
        return exists


def run_local(name, model_uri, flavor=None, config=None):
    raise NotImplementedError("run_local has not been implemented yet")

def target_help():
    help_msg = ("\nmlflow-triton plugin integrates the Triton Inference Server to the mlflow deployment pipeline. \n\n "  
                   "Example command: \n\n"
                   "  mlflow deployments create -t triton --name mymodel --flavor onnx -m models:/mymodel/Production -C \"version=1\" \n\n"
                  
                   "The environment variable TRITON_MODEL_REPO must be set to the location that the Triton"
                   "Inference Server is storing its models\n\n"

                   "export TRITON_MODEL_REPO = /path/to/triton/model/repo\n\n"                   
                   
                   "Use the following config options:\n\n"
                   
                   "- version: The version of the model to be released. This config will be used by Triton to create a new model sub-directory.\n")
    return help_msg
