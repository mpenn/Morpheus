import os


class Config(dict):
    def __init__(self):
        super().__init__()
        self['triton_url'] = os.environ.get('TRITON_URL')
        self['triton_model_repo'] = os.environ.get('TRITON_MODEL_REPO')
