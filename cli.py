from click.decorators import option
import configargparse
from config import Config

DEFAULT_CONFIG = Config.default()

if __name__ == '__main__':
    p = configargparse.ArgParser(default_config_files=['/etc/app/conf.d/*.conf', '~/.my_settings'])
    p.add('-c', '--my-config', is_config_file=True, help='config file path')

    pipeline_group = p.add_argument_group(title="pipeline", description="Options related to the model")
    pipeline_group.add_argument("--pipeline",
                                type=str,
                                choices=["triton", "pytorch", "tensorrt"],
                                default=DEFAULT_CONFIG.general.pipeline,
                                env_var='CLX_INFERENCE_PIPELINE')

    model_group = p.add_argument_group(title="model", description="Options related to the model")
    model_group.add_argument("--vocab_hash_file", type=str, default=DEFAULT_CONFIG.model.vocab_hash_file, env_var='CLX_MODEL_VOCAB_HASH_FILE')
    model_group.add_argument("--seq_length", type=int, default=DEFAULT_CONFIG.model.seq_length, env_var='CLX_MODEL_SEQ_LENGTH')
    model_group.add_argument("--max_batch_size", type=int, default=DEFAULT_CONFIG.model.max_batch_size, env_var='CLX_MODEL_MAX_BATCH_SIZE')

    kafka_group = p.add_argument_group(title="kafka", description="Options related to Kafka")
    kafka_group.add_argument("--bootstrap_servers", type=str, default=DEFAULT_CONFIG.kafka.bootstrap_servers, env_var='CLX_KAFKA_BOOTSTRAP_SERVERS')
    kafka_group.add_argument("--input_topic", type=str, default=DEFAULT_CONFIG.kafka.input_topic, env_var='CLX_KAFKA_INPUT_TOPIC')
    kafka_group.add_argument("--output_topic", type=str, default=DEFAULT_CONFIG.kafka.output_topic, env_var='CLX_KAFKA_OUTPUT_TOPIC')

    options = p.parse_args()

    # Now push this into the Config
    c = Config.get()

    c.general.pipeline = options.pipeline

    c.model.vocab_hash_file = options.vocab_hash_file
    c.model.seq_length = options.seq_length
    c.model.max_batch_size = options.max_batch_size

    c.kafka.bootstrap_servers = options.bootstrap_servers
    c.kafka.input_topic = options.input_topic
    c.kafka.output_topic = options.output_topic

    print("Config:")
    print(c.to_string())

    from run_pipeline import run_pipeline

    run_pipeline()
