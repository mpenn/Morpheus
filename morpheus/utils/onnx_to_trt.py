import typing

import tensorrt as trt

from morpheus.config import ConfigOnnxToTRT

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)


def gen_engine(c: ConfigOnnxToTRT):

    input_model = c.input_model

    print("Loading ONNX file: '{}'".format(input_model))

    # Otherwise we are creating a new model
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(input_model, "rb") as model_file:
            if (not parser.parse(model_file.read())):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                raise Exception("Count not parse Onnx file. See log.")

        # Now we need to build and serialize the model
        with builder.create_builder_config() as builder_config:

            builder_config.max_workspace_size = c.max_workspace_size * (1024 * 1024)
            builder_config.set_flag(trt.BuilderFlag.FP16)

            # Create the optimization files
            for min_batch, max_batch in c.batches:
                profile = builder.create_optimization_profile()

                min_shape = (min_batch, c.seq_length)
                shape = (max_batch, c.seq_length)

                for i in range(network.num_inputs):
                    in_tensor = network.get_input(i)
                    profile.set_shape(in_tensor.name, min=min_shape, opt=shape, max=shape)

                builder_config.add_optimization_profile(profile)

            # Actually build the engine
            print("Building engine. This may take a while...")
            engine = builder.build_engine(network, builder_config)

            # Now save a copy to prevent building next time
            print("Writing engine to: {}".format(c.output_model))
            serialized_engine = engine.serialize()

            with open(c.output_model, "wb") as f:
                f.write(serialized_engine)

            print("Complete!")
