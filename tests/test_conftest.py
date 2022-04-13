import pytest

from morpheus.config import Config
from morpheus.config import CppConfig


@pytest.mark.use_python
def test_mark_no_cpp(config: Config):
    assert not CppConfig.should_use_cpp, "Incorrect use_cpp"


@pytest.mark.use_cpp
def test_mark_only_cpp(config: Config):
    assert CppConfig.should_use_cpp, "Incorrect use_cpp"


def test_mark_neither(config: Config):
    pass


def test_explicit_fixture_no_cpp(config_no_cpp: Config):
    assert not CppConfig.should_use_cpp, "Incorrect use_cpp"


def test_explicit_fixture_only_cpp(config_only_cpp: Config):
    assert CppConfig.should_use_cpp, "Incorrect use_cpp"


class TestNoMarkerClass:

    def test_no_marker(self, config: Config):
        pass

    @pytest.mark.use_python
    def test_python_marker(self, config: Config):
        assert not CppConfig.should_use_cpp

    @pytest.mark.use_cpp
    def test_cpp_marker(self, config: Config):
        assert CppConfig.should_use_cpp

    @pytest.mark.slow
    def test_other_marker(self, config: Config):
        pass


@pytest.mark.use_python
class TestPythonMarkerClass:

    def test_no_marker(self, config: Config):
        assert not CppConfig.should_use_cpp

    @pytest.mark.use_python
    def test_extra_marker(self, config: Config):
        assert not CppConfig.should_use_cpp


@pytest.mark.use_cpp
class TestCppMarkerClass:

    def test_no_marker(self, config: Config):
        assert CppConfig.should_use_cpp

    @pytest.mark.use_cpp
    def test_extra_marker(self, config: Config):
        assert CppConfig.should_use_cpp
