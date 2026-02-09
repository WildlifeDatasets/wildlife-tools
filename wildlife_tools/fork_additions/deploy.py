import torch
import os
from onnxconverter_common import float16
import onnx

from wildlife_tools.fork_additions import (
    print_info,
)


def deploy_model(config, model):
    save_dir = config.save_directory
    dummy_input = torch.randn(1, 3, 224, 224)

    use_fp16 = False
    precision = "FP32"
    if config.device == "cuda" and torch.cuda.is_available():
        device_capability = torch.cuda.get_device_capability()
        if device_capability[0] >= 7:  # The hardare is new enough
            use_fp16 = True
            precision = "FP16"

    onnx_path = os.path.join(save_dir, "precision_track_re-identificator.onnx")

    print_info(f"Exporting model to ONNX format at '{onnx_path}'...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    if use_fp16:
        try:
            print_info("FP16 is supported on this hardware. Converting ONNX model to FP16...")
            model_onnx = onnx.load(onnx_path)
            model_fp16 = float16.convert_float_to_float16(model_onnx)
            onnx.save(model_fp16, onnx_path)
            print_info(f"Conversion to FP16 done.")
        except Exception as e:
            print_info(f"Error during FP16 conversion: {str(e)}.")

    print_info(f"ONNX export completed successfully. Your deployed ONNX checkpoint is available at '{onnx_path}'.")

    if config.device == "cuda":
        try:
            import tensorrt as trt

            print_info("CUDA is available. Exporting model to TensorRT format.")

            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, TRT_LOGGER)

            with open(onnx_path, "rb") as model_file:
                if not parser.parse(model_file.read()):
                    print_info("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        print_info(parser.get_error(error))
                    return

            config_trt = builder.create_builder_config()
            config_trt.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 3 << 30)

            profile = builder.create_optimization_profile()
            input_shapes = {
                "input": {
                    "min_shape": (1, 3, 224, 224),
                    "opt_shape": (1, 3, 224, 224),
                    "max_shape": (100, 3, 224, 224),
                }
            }

            for input_name, param in input_shapes.items():
                min_shape = param["min_shape"]
                opt_shape = param["opt_shape"]
                max_shape = param["max_shape"]
                profile.set_shape(input_name, min_shape, opt_shape, max_shape)

            if config_trt.add_optimization_profile(profile) < 0:
                print_info(f"Invalid optimization profile {profile}.")
                return

            precision = "FP32"
            if builder.platform_has_fast_fp16:
                config_trt.set_flag(trt.BuilderFlag.FP16)
                print_info("FP16 mode enabled for TensorRT.")
                precision = "FP16"

            serialized_engine = builder.build_serialized_network(network, config_trt)

            device_id = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(device_id).replace(" ", "")
            file_path, _ = os.path.splitext(os.path.basename(onnx_path))
            trt_basename = f"{file_path}_{gpu_name}_{precision}.engine"
            trt_path = os.path.join(save_dir, trt_basename)
            with open(trt_path, "wb") as f:
                f.write(serialized_engine)

            print_info(f"TensorRT engine saved to '{trt_path}'.")

        except ImportError:
            print_info("TensorRT or PyCUDA not installed. Skipping TensorRT deployment.")
            print_info("To enable TensorRT export, install: pip install tensorrt pycuda")
        except Exception as e:
            print_info(f"Error during TensorRT deployment: {str(e)}")
    else:
        print_info("CUDA not available. Skipping TensorRT deployment.")
