# Copyright (c) 2021-2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import logging.config
import logging.handlers
import multiprocessing
import os

import appdirs
import click
from srf.core import log_handler as srf_log_handler
from srf.core import logging as srf_logging
from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):

    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

        self._stdout = click.get_text_stream('stdout')
        self._stderr = click.get_text_stream('stderr')

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)

            is_error = record.levelno >= logging.ERROR

            file = self._stderr if is_error else self._stdout

            color_kwargs = self._determine_color(record.levelno)

            with tqdm.external_write_mode(file=file, nolock=False):
                # Write the message
                click.echo(click.style(msg, **color_kwargs), file=file, err=is_error)
                self.flush()

        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)

    def _determine_color(self, levelno: int):
        if (levelno >= logging.CRITICAL):
            return {"fg": "red", "bold": True}
        elif (levelno >= logging.ERROR):
            return {"fg": "red"}
        elif (levelno >= logging.WARNING):
            return {"fg": "yellow"}
        elif (levelno >= logging.INFO):
            return {}
        else:
            return {"dim": True}


def _configure_from_log_file(log_config_file: str):
    assert log_config_file is not None, "Log config file must be specified"

    ext = os.path.splitext(log_config_file)[1].lower()

    if (ext == "json"):

        dict_config: dict = None

        # Try and load from dict
        with open(log_config_file, "r") as fp:
            dict_config = json.load(fp)

        logging.config.dictConfig(dict_config)
    else:
        # Must be another ini type file
        logging.config.fileConfig(log_config_file)


def _configure_from_log_level(log_level: int):
    """
    Default config with only option being the logging level. Outputs to both the console and a file. Sets up a logging
    producer/consumer that works well in multi-thread/process environments.

    Parameters
    ----------
    log_level : int
        Log level and above to report
    """
    # Default config with level
    logging.captureWarnings(True)

    # Get the root Morpheus logger
    srf_logging.init_logging("morpheus", log_level)
    morpheus_logger = logging.getLogger("morpheus")
    morpheus_logger.setLevel(log_level)

    # Dont propagate upstream
    morpheus_logger.propagate = False
    morpheus_logging_queue = multiprocessing.Queue()

    # This needs the be the only handler for morpheus logger
    morpheus_queue_handler = logging.handlers.QueueHandler(morpheus_logging_queue)

    # At this point, any morpheus logger will propagate upstream to the morpheus root and then be handled by the queue
    # handler
    morpheus_logger.addHandler(morpheus_queue_handler)

    log_file = os.path.join(appdirs.user_log_dir(appauthor="NVIDIA", appname="morpheus"), "morpheus.log")

    # Ensure the log directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Now we build all of the handlers for the queue listener
    file_handler = logging.handlers.RotatingFileHandler(filename=log_file, backupCount=5, maxBytes=1000000)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - [%(levelname)s]: %(message)s {%(name)s, %(threadName)s}'))

    # Tqdm stream handler (avoids messing with progress bars)
    #console_handler = TqdmLoggingHandler()
    console_handler = srf_log_handler.SrfHandler()

    # Build and run the queue listener to actually process queued messages
    queue_listener = logging.handlers.QueueListener(morpheus_logging_queue,
                                                    console_handler,
                                                    file_handler,
                                                    respect_handler_level=True)
    queue_listener.start()
    queue_listener._thread.name = "Logging Thread"

    # Register a function to kill the listener thread before shutting down. prevents error on intpreter close
    def stop_queue_listener():
        queue_listener.stop()

    import atexit
    atexit.register(stop_queue_listener)


def configure_logging(log_level: int, log_config_file: str = None):
    """
    Configures Morpheus logging in one of two ways. Either specifying a logging config file to load or a logging level
    which will use a default configuration. The default configuration outputs to both the console and a file. Sets up a
    logging producer/consumer that works well in multi-thread/process environments.

    Parameters
    ----------
    log_level: int
        Specifies the log level and above to output. Must be one of the available levels in the `logging` module.
    log_config_file: str, optional (default = None):
        Instructs Morpheus to configure logging via a config file. These config files can be complex and are outlined in
        the Python logging documentation. Will accept either a ``.ini`` file which will be loaded via
        `logging.config.fileConfig()` (See `here
        <https://docs.python.org/3/library/logging.config.html#logging.config.fileConfig>`__) or a ``.json`` file which
        will be loaded via `logging.config.dictConfig()` (See `here
        <https://docs.python.org/3/library/logging.config.html#logging.config.dictConfig>`__). Defaults to None.
    """

    if (log_config_file is not None):
        # Configure using log file
        _configure_from_log_file(log_config_file=log_config_file)
    else:
        _configure_from_log_level(log_level=log_level)


def deprecated_stage_warning(logger, cls, name):
    logger.warning(("The '%s' stage ('%s') is no longer required to manage backpressure and has been deprecated. "
                    "It has no effect and acts as a pass through stage."),
                   cls.__name__,
                   name)
