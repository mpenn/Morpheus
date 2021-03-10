import click
from config import Config

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

@click.group(chain=True, cls=DefaultGroup, invoke_without_command=True)
@click.option('--debug/--no-debug', default=False)
@click.option('--pipeline',
              default=DEFAULT_CONFIG.general.pipeline,
              type=click.Choice(["triton", "pytorch", "tensorrt"], case_sensitive=False),
              help="")
@click.pass_context
def cli(ctx: click.Context, **kwargs):

    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below
    ctx.ensure_object(dict)

    kwargs = _without_empty_args(kwargs)

    c = Config.get().general

    for param in kwargs:
        if hasattr(c, param):
            setattr(c, param, kwargs[param])

    pass


@click.group(invoke_without_command=True)
@click.pass_context
def cli2(ctx: click.Context, **kwargs):

    ctx.ensure_object(dict)

    pass

@cli.command()
@click.option('--vocab_hash_file', default=DEFAULT_CONFIG.model.vocab_hash_file, type=click.Path(exists=True, dir_okay=False), help="")
@click.option('--seq_length', default=DEFAULT_CONFIG.model.seq_length, type=click.IntRange(min=1), help="")
@click.option('--max_batch_size', default=DEFAULT_CONFIG.model.max_batch_size, type=click.IntRange(min=1), help="")
@click.pass_context
def model(ctx: click.Context, **kwargs):

    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below
    ctx.ensure_object(dict)

    kwargs = _without_empty_args(kwargs)

    c = Config.get().model

    for param in kwargs:
        if hasattr(c, param):
            setattr(c, param, kwargs[param])

    pass


@cli.command()
@click.option('--bootstrap_servers', default=DEFAULT_CONFIG.kafka.bootstrap_servers, type=str, help="")
@click.option('--input_topic', default=DEFAULT_CONFIG.kafka.input_topic, type=str, help="")
@click.option('--output_topic', default=DEFAULT_CONFIG.kafka.output_topic, type=str, help="")
@click.pass_context
def kafka(ctx: click.Context, **kwargs):

    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below
    ctx.ensure_object(dict)

    kwargs = _without_empty_args(kwargs)

    c = Config.get().kafka

    for param in kwargs:
        if hasattr(c, param):
            setattr(c, param, kwargs[param])

    pass


@cli.command("run")
@click.pass_context
def do_run(ctx: click.Context, **kwargs):

    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below
    ctx.ensure_object(dict)

    kwargs = _without_empty_args(kwargs)

    from run_pipeline import run_pipeline

    run_pipeline()

all_cli = click.CommandCollection(sources=[cli, cli2])

if __name__ == '__main__':
    cli(obj={}, auto_envvar_prefix='CLX', show_default=True)

    print("Config: ")
    print(Config.get().to_string())

    # run_asyncio_loop()
    from run_pipeline import run_pipeline

    run_pipeline()