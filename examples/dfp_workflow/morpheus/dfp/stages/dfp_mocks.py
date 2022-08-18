import json
import random
import typing

from collections import defaultdict
from datetime import date
from datetime import datetime

import boto3
import numpy as np
import srf

import cudf

from .dfp_autoencoder import DFPAutoEncoder

from morpheus.messages import MessageMeta
from morpheus.messages.message_meta import UserMessageMeta
from morpheus.messages.multi_ae_message import MultiAEMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair


def get_dummy_json():
    json_data_entry = [{
        "access_device": {
            "ip":
                f"{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}",
            "is_encryption_enabled":
                f"{random.choice([True, False])}",
            "is_firewall_enabled":
                f"{random.choice([True, False])}",
            "is_password_set":
                f"{random.choice([True, False])}",
            "location": {
                "city": "Some city", "country": "Some country", "state": "Some state"
            }
        },
        "alias": "Some alias",
        "application": {
            "key": "app_id", "name": "my_application"
        },
        "auth_device": {
            "location": {}, "name": "authorizing device"
        },
        "browser": f"{random.choice(['Firefox', 'Chrome', 'Edge', 'Safari'])}",
        "os": f"{random.choice(['Linux', 'OSX', 'Android', 'Windows'])}",
        "email": "john.doe@gmail.com",
        "event_type": "Some event",
        "factor": "42",
        "isotimestamp": f"{datetime.isoformat(datetime.now())}",
        "reason": "yes",
        "result": "no",
        "timestamp": datetime.timestamp(datetime.now()),
        "txid": f"{random.randint(0, 1000000000)}",
        "user": {
            "groups": ["group1", "group2"], "key": "user_key", "name": f"User_{chr(random.randint(65, 90))}"
        }
    }]

    return json_data_entry


def s3_object_generator_mock_duo():
    max_items = 1000
    s3_session = boto3.Session()
    s3_resource_handle = s3_session.resource('s3')
    current_date = date.today()

    for idx in range(max_items):
        object_key = f'DUO_AUTH_{current_date:%Y}-{current_date:%m}-{current_date:%d}'
        s3_dummy_object = s3_resource_handle.Object("DUO_MOCK_BUCKET", object_key)
        setattr(s3_dummy_object, "DEBUG_JSON", json.dumps(get_dummy_json()))

        yield s3_dummy_object


user_index_count_mock = defaultdict(lambda: 0)


def s3_writer_mock(message: cudf.DataFrame):
    if (message is None):
        return message

    # global user_index_count_mock

    current_date = datetime.today()
    object_suffix = f'DUO_AUTH_{current_date:%Y}-{current_date:%m}-{current_date:%d}'

    # dict_df = message.to_dict(orient="records")
    # user = message.get_user_id()
    # idx_current = user_index_count_mock[user]

    for index, record in message.iterrows():

        record_dict = record.to_dict()

        record_dict["timestamp"] = record["timestamp"].isoformat()

        record_key = f"hammah_duo_result_{record['username']}_{index}_{object_suffix}.json"
        print(f"WRITER MOCK: Writing to {record_key}\n{json.dumps(record_dict)}")


class DFPDuoInferenceMock(SinglePortStage):

    @property
    def name(self) -> str:
        return "dfp-inference-mock"

    def supports_cpp_node(self):
        return False

    def accepted_types(self) -> typing.Tuple:
        return (MessageMeta, )

    def on_data(self, message: UserMessageMeta):
        if (not message or message.df.empty):
            return None

        df_user = message.df
        user = message.user_id

        output_message = MultiAEMessage(message, mess_offset=0, mess_count=message.count, model=DFPAutoEncoder())
        output_message.set_meta('anomaly_score', cudf.Series(np.random.uniform(0.0, 1.0, df_user.shape[0])))
        output_message.set_meta('model_version', 'none')

        return output_message

    def _build_single(self, builder: srf.Builder, input_stream: StreamPair) -> StreamPair:
        node = builder.make_node(self.unique_name, self.on_data)
        builder.make_edge(input_stream[0], node)

        return node, MultiAEMessage
