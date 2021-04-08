import click
from config import Config, ConfigOnnxToTRT, auto_determine_bootstrap
from pipeline import Pipeline

DEFAULT_CONFIG = Config.default()


def _without_empty_args(passed_args):
    return {k: v for k, v in passed_args.items() if v is not None}


class DefaultGroup(click.Group):
    def resolve_command(self, ctx, args):
        base = super(DefaultGroup, self)
        cmd_name, cmd, args = base.resolve_command(ctx, args)
        if hasattr(ctx, 'arg0'):
            args.insert(0, ctx.arg0)
            cmd_name = cmd.name
        return cmd_name, cmd, args


@click.group(chain=False, invoke_without_command=True)
@click.option('--debug/--no-debug', default=False)
@click.pass_context
def cli(ctx: click.Context, **kwargs):

    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below
    ctx.ensure_object(dict)

    kwargs = _without_empty_args(kwargs)

    c = Config.get()

    for param in kwargs:
        if hasattr(c, param):
            setattr(c, param, kwargs[param])


@cli.command(short_help="Converts an ONNX model to a TRT engine")
@click.option("--input_model", type=click.Path(exists=True, readable=True), required=True)
@click.option("--output_model", type=click.Path(exists=False, writable=True), required=True)
@click.option('--batches', type=(int, int), required=True, multiple=True)
@click.option('--seq_length', type=int, required=True)
@click.option('--max_workspace_size', type=int, default=16000)
@click.pass_context
def onnx_to_trt(ctx: click.Context, **kwargs):

    print("Generating onnx file")

    # Convert batches to a list
    kwargs["batches"] = list(kwargs["batches"])

    kwargs = _without_empty_args(kwargs)

    c = ConfigOnnxToTRT()

    for param in kwargs:
        if hasattr(c, param):
            setattr(c, param, kwargs[param])

    from onnx_to_trt import gen_engine

    gen_engine(c)


@cli.group(chain=True, invoke_without_command=True, short_help="Run the inference pipeline")
@click.option('--num_threads',
              default=DEFAULT_CONFIG.num_threads,
              type=click.IntRange(min=1),
              help="Number of internal pipeline threads to use")
@click.option(
    '--pipeline_batch_size',
    default=DEFAULT_CONFIG.pipeline_batch_size,
    type=click.IntRange(min=1),
    help="Internal batch size for the pipeline. Can be much larger than the model batch size. Also used for Kafka consumers")
@click.option('--model_vocab_hash_file',
              default=DEFAULT_CONFIG.model_vocab_hash_file,
              type=click.Path(exists=True, dir_okay=False),
              help="Model vocab file to use for pre-processing")
@click.option('--model_seq_length',
              default=DEFAULT_CONFIG.model_seq_length,
              type=click.IntRange(min=1),
              help="Sequence length to use for the model")
@click.option('--model_max_batch_size',
              default=DEFAULT_CONFIG.model_max_batch_size,
              type=click.IntRange(min=1),
              help="Max batch size to use for the model")
@click.pass_context
def pipeline(ctx: click.Context, **kwargs):
    """Configure and run the pipeline. To configure the pipeline, list the stages in the order that data should flow. The output of each stage will become the input for the next stage. For example, to read, classify and write to a file, the following stages could be used

    \b
    pipeline from-file --filename=my_dataset.json deserialize preprocess inf-triton --model_name=my_model --server_url=localhost:8001 filter --threshold=0.5 to-file --filename=classifications.json

    \b
    Pipelines must follow a few rules:
    1. Data must originate in a source stage. Current options are `from-file` or `from-kafka`
    2. A `deserialize` stage must be placed between the source stages and the rest of the pipeline
    3. Only one inference stage can be used. Zero is also fine
    4. The following stages must come after an inference stage: `add-class`, `filter`, `gen-viz`

    """

    print("Building pipeline")

    kwargs = _without_empty_args(kwargs)

    c = Config.get()

    for param in kwargs:
        if hasattr(c, param):
            setattr(c, param, kwargs[param])

    ctx.obj = Pipeline(c)

    return ctx.obj


@pipeline.resultcallback()
@click.pass_context
def post_pipeline(ctx: click.Context, stages, **kwargs):

    print("Running pipeline... Ctrl+C to Quit")

    p: Pipeline = ctx.ensure_object(Pipeline)

    # Run the pipeline
    p.run()


@pipeline.command(short_help="Load messages from a file")
@click.option('--filename', type=click.Path(exists=True, dir_okay=False), help="Input filename")
@click.pass_context
def from_file(ctx: click.Context, **kwargs):

    p: Pipeline = ctx.ensure_object(Pipeline)

    from pipeline import FileSourceStage2

    stage = FileSourceStage2(Config.get(), **kwargs)

    p.set_source(stage)

    return stage


@pipeline.command(short_help="Load messages from a Kafka cluster")
@click.option(
    '--bootstrap_servers',
    type=str,
    default="auto",
    required=True,
    help=
    "Comma-separated list of bootstrap servers. If using Kafka created via `docker-compose`, this can be set to 'auto' to automatically determine the cluster IPs and ports"
)
@click.option('--input_topic', type=str, default="test_pcap", required=True, help="Kafka topic to read from")
@click.option('--group_id', type=str, default="custreamz", required=True, help="")
@click.option('--use_dask', is_flag=True, help="Whether or not to use dask for multiple processes reading from Kafka")
@click.option('--poll_interval',
              type=str,
              default="10millis",
              required=True,
              help="Polling interval to check for messages. Follows the pandas interval format")
# @click.option('--max_batch_size', type=int, default=1000, required=True, help="Maximum messages that can be pulled off the server at a time. Should ")
@click.pass_context
def from_kafka(ctx: click.Context, **kwargs):

    p: Pipeline = ctx.ensure_object(Pipeline)

    kwargs = _without_empty_args(kwargs)

    if ("bootstrap_servers" in kwargs and kwargs["bootstrap_servers"]):
        kwargs["bootstrap_servers"] = auto_determine_bootstrap()

    from pipeline import KafkaSourceStage

    stage = KafkaSourceStage(Config.get(), **kwargs)

    p.set_source(stage)

    return stage


@pipeline.command(short_help="Display throughput numbers at a specific point in the pipeline")
@click.option('--description', type=str, required=True, help="Header message to use for this monitor")
@click.option('--smoothing',
              type=float,
              default=0.05,
              help="How much to average throughput numbers. 0=full average, 1=instantaneous")
@click.option('--unit', type=str, help="Units to use for data rate")
@click.pass_context
def monitor(ctx: click.Context, **kwargs):

    p: Pipeline = ctx.ensure_object(Pipeline)

    kwargs = _without_empty_args(kwargs)

    from pipeline import TqdmStage

    stage = TqdmStage(Config.get(), **kwargs)

    p.add_stage(stage)

    return stage


@pipeline.command(short_help="Buffer results")
@click.option('--count', type=int, default=1000, help="")
@click.pass_context
def buffer(ctx: click.Context, **kwargs):

    p: Pipeline = ctx.ensure_object(Pipeline)

    kwargs = _without_empty_args(kwargs)

    from pipeline import BufferStage

    stage = BufferStage(Config.get(), **kwargs)

    p.add_stage(stage)

    return stage


@pipeline.command(short_help="Deserialize source data from JSON")
@click.pass_context
def deserialize(ctx: click.Context, **kwargs):

    p: Pipeline = ctx.ensure_object(Pipeline)

    kwargs = _without_empty_args(kwargs)

    from pipeline import DeserializeStage

    stage = DeserializeStage(Config.get(), **kwargs)

    p.add_stage(stage)

    return stage


@pipeline.command(short_help="Convert messages to tokens")
@click.pass_context
def preprocess(ctx: click.Context, **kwargs):

    p: Pipeline = ctx.ensure_object(Pipeline)

    kwargs = _without_empty_args(kwargs)

    from pipeline import PreprocessStage

    stage = PreprocessStage(Config.get(), **kwargs)

    p.add_stage(stage)

    return stage


@pipeline.command(short_help="Perform inference with Triton")
@click.option('--model_name', type=str, required=True, help="Model name in Triton to send messages to")
@click.option('--server_url', type=str, required=True, help="Triton server URL (IP:Port)")
@click.pass_context
def inf_triton(ctx: click.Context, **kwargs):

    p: Pipeline = ctx.ensure_object(Pipeline)

    kwargs = _without_empty_args(kwargs)

    from inference_triton import TritonInferenceStage

    stage = TritonInferenceStage(Config.get(), **kwargs)

    p.add_stage(stage)

    return stage


@pipeline.command(short_help="Add detected classifications to each message")
@click.option('--threshold', type=float, default=0.5, required=True, help="Level to consider True/False")
@click.pass_context
def add_class(ctx: click.Context, **kwargs):

    p: Pipeline = ctx.ensure_object(Pipeline)

    kwargs = _without_empty_args(kwargs)

    from pipeline import AddClassificationsStage

    stage = AddClassificationsStage(Config.get(), **kwargs)

    p.add_stage(stage)

    return stage


@pipeline.command(short_help="Filter message by a classification threshold")
@click.option('--threshold', type=float, default=0.5, required=True, help="")
@click.pass_context
def filter(ctx: click.Context, **kwargs):

    p: Pipeline = ctx.ensure_object(Pipeline)

    kwargs = _without_empty_args(kwargs)

    from pipeline import FilterDetectionsStage

    stage = FilterDetectionsStage(Config.get(), **kwargs)

    p.add_stage(stage)

    return stage


@pipeline.command(short_help="Write all messages to a file")
@click.option('--filename', type=click.Path(writable=True), required=True, help="")
@click.option('--overwrite', is_flag=True, help="")
@click.pass_context
def to_file(ctx: click.Context, **kwargs):

    p: Pipeline = ctx.ensure_object(Pipeline)

    kwargs = _without_empty_args(kwargs)

    from pipeline import WriteToFileStage

    stage = WriteToFileStage(Config.get(), **kwargs)

    p.add_stage(stage)

    return stage


@pipeline.command(short_help="Write all messages to a Kafka cluster")
@click.option(
    '--bootstrap_servers',
    type=str,
    default="auto",
    required=True,
    help=
    "Comma-separated list of bootstrap servers. If using Kafka created via `docker-compose`, this can be set to 'auto' to automatically determine the cluster IPs and ports"
)
@click.option('--output_topic', type=str, required=True, help="Output Kafka topic to publish to")
@click.pass_context
def to_kafka(ctx: click.Context, **kwargs):

    p: Pipeline = ctx.ensure_object(Pipeline)

    kwargs = _without_empty_args(kwargs)

    if ("bootstrap_servers" in kwargs and kwargs["bootstrap_servers"]):
        kwargs["bootstrap_servers"] = auto_determine_bootstrap()

    from pipeline import WriteToKafkaStage

    stage = WriteToKafkaStage(Config.get(), **kwargs)

    p.add_stage(stage)

    return stage


@pipeline.command(short_help="Write out vizualization data frames")
@click.option('--out_dir', type=click.Path(dir_okay=True, file_okay=False), default="./viz_frames", required=True, help="")
@click.option('--overwrite', is_flag=True, help="")
@click.pass_context
def gen_viz(ctx: click.Context, **kwargs):

    p: Pipeline = ctx.ensure_object(Pipeline)

    kwargs = _without_empty_args(kwargs)

    from pipeline import GenerateVizFramesStage

    stage = GenerateVizFramesStage(Config.get(), **kwargs)

    p.add_stage(stage)

    return stage


if __name__ == '__main__':
    cli(obj={}, auto_envvar_prefix='CLX', show_default=True)

    print("Config: ")
    print(Config.get().to_string())

    # run_asyncio_loop()
    from run_pipeline import run_pipeline

    run_pipeline()