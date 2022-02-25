import pytest

from morpheus.config import Config


@pytest.mark.use_python
def test_mark_no_cpp(config: Config):

    assert config.use_cpp is False, "Incorrect use_cpp"


@pytest.mark.use_cpp
def test_mark_only_cpp(config: Config):

    assert config.use_cpp is True, "Incorrect use_cpp"


def test_mark_neither(config: Config):

    pass


def test_explicit_fixture_no_cpp(config_no_cpp: Config):
    assert config_no_cpp.use_cpp is False, "Incorrect use_cpp"


def test_explicit_fixture_only_cpp(config_only_cpp: Config):
    assert config_only_cpp.use_cpp is True, "Incorrect use_cpp"


@pytest.mark.use_python
class TestPythonMarkerClass:
    def test_no_marker(self, config: Config):
        assert not config.use_cpp

    @pytest.mark.use_python
    def test_extra_marker(self, config: Config):
        assert not config.use_cpp


@pytest.mark.use_cpp
class TestCppMarkerClass:
    def test_no_marker(self, config: Config):
        assert config.use_cpp

    @pytest.mark.use_cpp
    def test_extra_marker(self, config: Config):
        assert config.use_cpp
