import pytest


def pytest_addoption(parser: pytest.Parser):
    """
    Adds command line options for running specfic tests that are disabled by default
    """
    parser.addoption(
        "--run_slow",
        action="store_true",
        dest="run_slow",
        help="Run slow tests that would otherwise be skipped",
    )


def pytest_generate_tests(metafunc: pytest.Metafunc):
    """
    This function will add parameterizations for the `config` fixture depending on what types of config the test
    supports
    """

    # Only care about the config fixture
    if ("config" not in metafunc.fixturenames):
        return

    use_cpp = metafunc.definition.get_closest_marker("use_cpp") is not None
    use_python = metafunc.definition.get_closest_marker("use_python") is not None

    if (use_cpp and use_python):
        raise RuntimeError(
            "Both markers (use_cpp and use_python) were added to function {}. Remove markers to support both.".format(
                metafunc.definition.nodeid))
    elif (not use_cpp and not use_python):
        # Add the markers to the parameters
        metafunc.parametrize("config",
                             [
                                 pytest.param(True, marks=pytest.mark.use_cpp, id="use_cpp"),
                                 pytest.param(False, marks=pytest.mark.use_python, id="use_python")
                             ],
                             indirect=True)


def pytest_runtest_setup(item):
    if (not item.config.getoption("--run_slow")):
        if (item.get_closest_marker("slow") is not None):
            pytest.skip("Skipping slow tests by default. Use --run_slow to enable")


def pytest_collection_modifyitems(items):
    """
    To support old unittest style tests, try to determine the mark from the name
    """
    for item in items:
        if "no_cpp" in item.nodeid:
            item.add_marker(pytest.mark.use_python)
        elif "cpp" in item.nodeid:
            item.add_marker(pytest.mark.use_cpp)


@pytest.fixture(scope="function")
def config_only_cpp():
    """
    Use this fixture in unittest style tests to indicate a lack of support for C++. Use via
    `@pytest.mark.usefixtures("config_only_cpp")`
    """

    from morpheus.config import Config

    config = Config.get()

    config.use_cpp = True

    return config


@pytest.fixture(scope="function")
def config_no_cpp():
    """
    Use this fixture in unittest style tests to indicate support for C++. Use via
    `@pytest.mark.usefixtures("config_no_cpp")`
    """

    from morpheus.config import Config

    config = Config.get()

    config.use_cpp = False

    return config


@pytest.fixture(scope="function")
def config(request: pytest.FixtureRequest):
    """
    For new pytest style tests, get the config by using this fixture. It will setup the config based on the marks set on
    the object. If no marks are added to the test, it will be parameterized for both C++ and python. For example,

    ```
    @pytest.mark.use_python
    def my_python_test(config: Config):
        ...
    ```
    """

    from morpheus.config import Config

    config = Config.get()

    if (not hasattr(request, "param")):
        use_cpp = request.node.get_closest_marker("use_cpp") is not None
        use_python = request.node.get_closest_marker("use_python") is not None

        assert use_cpp != use_python, "Invalid config"

        if (use_cpp):
            config.use_cpp = True
        else:
            config.use_cpp = False

    else:
        if (request.param):
            config.use_cpp = True
        else:
            config.use_cpp = False

    return config
