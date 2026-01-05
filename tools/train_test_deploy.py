from itertools import chain
import torch
import timm
from timm import create_model
import pandas as pd
import os
import torchvision.transforms as T
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from torch.optim import SGD
import yaml
from addict import Dict
from sklearn.metrics import confusion_matrix
from onnxconverter_common import float16
import onnx

from wildlife_tools.data import WildlifeDataset
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.inference import KnnClassifier
from wildlife_tools.train import ArcFaceLoss
from wildlife_tools.fork_additions import (
    BasicTrainerWithValidation,
    create_labels_file,
    cuda_available,
    warn_confused_pairs,
    print_info,
)


def train(config):
    labels_path = os.path.abspath(os.path.join(config.dataset_directory, "labels.csv"))
    if os.path.exists(labels_path) and os.path.isfile(labels_path):
        print_info(
            f"Labels file '{labels_path}' found. Will NOT be creating a new one. If you wish to update your labels file, simply delete the existing one."
        )
    else:
        print_info(f"Labels file '{labels_path}' not found. Will be creating a new one.")
        create_labels_file(config.dataset_directory, config.val_split, config.seed)

    metadata = pd.read_csv(labels_path)
    dataset = WildlifeDataset(
        metadata=metadata.query('split == "train"'), root=config.dataset_directory, transform=config.train_transforms
    )
    val_dataset = WildlifeDataset(
        metadata=metadata.query('split == "val"'), root=config.dataset_directory, transform=config.test_transforms
    )

    backbone = timm.create_model("hf-hub:BVRA/MegaDescriptor-T-224", num_classes=0, pretrained=True)
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224)
        embedding_size = backbone(dummy_input).shape[1]
    objective = ArcFaceLoss(num_classes=dataset.num_classes, embedding_size=embedding_size, margin=0.5, scale=64)

    params = chain(backbone.parameters(), objective.parameters())
    optimizer = SGD(params=params, lr=0.001, momentum=0.9)
    min_lr = optimizer.defaults.get("lr") * 1e-3

    epochs = config.epochs
    warmup_epochs = int(2 * epochs / 3)
    cosine_epochs = epochs - warmup_epochs

    warmup = LambdaLR(optimizer, lr_lambda=lambda epoch: (epoch + 1) / warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=min_lr)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    trainer = BasicTrainerWithValidation(
        dataset=dataset,
        val_dataset=val_dataset,
        save_dir=config.dataset_directory,
        checkpoint_name="precision_track_re-identificator.pth",
        model=backbone,
        objective=objective,
        optimizer=optimizer,
        scheduler=scheduler,
        batch_size=config.batch_size,
        accumulation_steps=160 // config.batch_size,
        num_workers=2,
        epochs=epochs,
        device=config.device,
    )
    trainer.train()
    print_info("Done training.")


def test(config):
    model = create_model("hf-hub:BVRA/MegaDescriptor-t-224", pretrained=False)

    state_dict = torch.load(
        os.path.join(config.dataset_directory, "precision_track_re-identificator.pth"),
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(state_dict["model"])
    extractor = DeepFeatures(model, device=config.device, batch_size=30)

    metadata = pd.read_csv(os.path.join(config.dataset_directory, "labels.csv"), index_col=0)

    database = WildlifeDataset(
        metadata=metadata.query('split == "train"'),
        root=config.dataset_directory,
        transform=config.test_transforms,
    )

    query = WildlifeDataset(
        metadata=metadata.query('split == "val"'),
        root=config.dataset_directory,
        transform=config.test_transforms,
    )

    matcher = CosineSimilarity()
    similarity = matcher(query=extractor(query), database=extractor(database))
    preds = KnnClassifier(k=1, database_labels=database.labels_string)(similarity)

    unique_labels = sorted(set(query.labels_string) | set(preds))
    conf_matrix = confusion_matrix(query.labels_string, preds, labels=unique_labels)
    conf_matrix_df = pd.DataFrame(conf_matrix, index=unique_labels, columns=unique_labels)
    conf_matrix_df.to_csv(os.path.join(config.dataset_directory, "re-identification_confusion_matrix.csv"))

    warn_confused_pairs(conf_matrix, unique_labels)
    print_info("Done testing.")


def deploy(config):
    model = create_model("hf-hub:BVRA/MegaDescriptor-t-224", pretrained=False)

    state_dict = torch.load(
        os.path.join(config.dataset_directory, "precision_track_re-identificator.pth"),
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(state_dict["model"])
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    use_fp16 = False
    precision = "FP32"
    if config.device == "cuda" and torch.cuda.is_available():
        device_capability = torch.cuda.get_device_capability()
        if device_capability[0] >= 7:  # The hardare is new enough
            use_fp16 = True
            precision = "FP16"

    onnx_path = os.path.join(config.dataset_directory, "precision_track_re-identificator_DEPLOYED.onnx")

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
            trt_path = os.path.join(config.dataset_directory, trt_basename)
            with open(trt_path, "wb") as f:
                f.write(serialized_engine)

            print_info(f"TensorRT engine saved to '{trt_path}'.")

        except ImportError:
            print_info("TensorRT or PyCUDA not installed. Skipping TensorRT deployment.")
            print_info("To enable TensorRT export, install: pip install tensorrt pycuda")
        # except Exception as e:
        #     print_info(f"Error during TensorRT deployment: {str(e)}")
    else:
        print_info("CUDA not available. Skipping TensorRT deployment.")

    print_info("Done deploying.")


def main():
    with open(os.path.join("..", "configs", "user_configs.yaml"), "r") as f:
        config = Dict(yaml.safe_load(f))

    if cuda_available():
        print_info("Your machine is CUDA accelerated. Therefore, the processes will take place on GPU.")
        config.device = "cuda"
    else:
        print_info("Your machine is NOT CUDA accelerated. Therefore, the processes will take place on CPU.")
        config.device = "cpu"

    config.train_transforms = T.Compose(
        [
            T.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
            T.RandAugment(num_ops=2, magnitude=20),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    config.test_transforms = T.Compose(
        [
            T.Resize(size=(224, 224)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    if config.train:
        print_info("Training...")
        train(config)
    else:
        print_info("Skipping training...")

    if config.test:
        print_info("Testing...")
        test(config)
    else:
        print_info("Skipping testing...")

    if config.deploy:
        print_info("Deploying")
        deploy(config)
    else:
        print_info("Skipping deploying...")


if __name__ == "__main__":
    main()
