import importlib
import os

import cupy as cp
import pytest

import morpheus._lib.messages as neom
from morpheus.pipeline import messages


def check_message(python_type: type, cpp_type: type, should_be_cpp: bool, no_cpp_class: bool, args: tuple):
    instance = python_type(*args)

    # Check that the C++ type is set in the class
    expected_cpp_class = None if no_cpp_class else cpp_type
    assert python_type._cpp_class is expected_cpp_class

    # Check that the isinstance to Python type works
    assert isinstance(instance, python_type)

    # Check that the instantiated class is the right type
    expected_class = cpp_type if should_be_cpp and cpp_type is not None else python_type
    assert instance.__class__ is expected_class


def check_all_messages(should_be_cpp: bool, no_cpp_class: bool):

    check_message(messages.MessageMeta, neom.MessageMeta, should_be_cpp, no_cpp_class, (None, ))

    # UserMessageMeta doesn't contain a C++ impl, so we should
    # always received the python impl
    check_message(messages.UserMessageMeta, None, should_be_cpp, no_cpp_class, (None, None))

    check_message(messages.MultiMessage, neom.MultiMessage, should_be_cpp, no_cpp_class, (None, 0, 1))

    assert messages.InferenceMemory._cpp_class is None if no_cpp_class else neom.InferenceMemory
    # C++ impl for InferenceMemory doesn't have a constructor
    if (should_be_cpp):
        pytest.raises(TypeError, messages.InferenceMemory, 1)

    cp_array = cp.zeros((1, 2))

    check_message(messages.InferenceMemoryNLP,
                  neom.InferenceMemoryNLP,
                  should_be_cpp,
                  no_cpp_class, (1, cp_array, cp_array, cp_array))

    check_message(messages.InferenceMemoryFIL,
                  neom.InferenceMemoryFIL,
                  should_be_cpp,
                  no_cpp_class, (1, cp_array, cp_array))

    # No C++ impl, should always get the Python class
    check_message(messages.InferenceMemoryAE, None, should_be_cpp, no_cpp_class, (1, cp_array, cp_array))

    check_message(messages.MultiInferenceMessage,
                  neom.MultiInferenceMessage,
                  should_be_cpp,
                  no_cpp_class, (None, 0, 1, None, 0, 1))

    check_message(messages.MultiInferenceNLPMessage,
                  neom.MultiInferenceNLPMessage,
                  should_be_cpp,
                  no_cpp_class, (None, 0, 1, None, 0, 1))

    check_message(messages.MultiInferenceFILMessage,
                  neom.MultiInferenceFILMessage,
                  should_be_cpp,
                  no_cpp_class, (None, 0, 1, None, 0, 1))

    assert messages.ResponseMemory._cpp_class is None if no_cpp_class else neom.ResponseMemory
    # C++ impl doesn't have a constructor
    if (should_be_cpp):
        pytest.raises(TypeError, messages.ResponseMemory, 1)

    check_message(messages.ResponseMemoryProbs, neom.ResponseMemoryProbs, should_be_cpp, no_cpp_class, (1, cp_array))

    # No C++ impl
    check_message(messages.ResponseMemoryAE, None, should_be_cpp, no_cpp_class, (1, cp_array))

    check_message(messages.MultiResponseMessage,
                  neom.MultiResponseMessage,
                  should_be_cpp,
                  no_cpp_class, (None, 0, 1, None, 0, 1))

    check_message(messages.MultiResponseProbsMessage,
                  neom.MultiResponseProbsMessage,
                  should_be_cpp,
                  no_cpp_class, (None, 0, 1, None, 0, 1))

    # No C++ impl
    check_message(messages.MultiResponseAEMessage, None, should_be_cpp, no_cpp_class, (None, 0, 1, None, 0, 1, ''))


def test_constructor_cpp(config):
    check_all_messages(config.use_cpp, False)


@pytest.mark.reload_modules(messages)
@pytest.mark.usefixtures("reload_modules", "restore_environ")
@pytest.mark.use_cpp
def test_constructor_env(config):
    os.environ['MORPHEUS_NO_CPP'] = '1'
    importlib.reload(messages)

    check_all_messages(False, True)
