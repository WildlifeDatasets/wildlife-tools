from itertools import chain
import os
import torchvision.transforms as T
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from torch.optim import SGD
import yaml
from addict import Dict

from wildlife_tools.fork_additions import (
    PtReIDModel,
    ClassificationTrainerWithValidation,
    BalancedImageDataset,
    create_labels_file,
    cuda_available,
    test_metrics,
    test_classification,
    print_info,
    deploy_model,
    ArcFaceWithCrossEntropyLoss,
    NumpyDataset,
)


def train(config):
    dataset = BalancedImageDataset(
        metadata=config.metadata,
        root=config.dataset_directory,
        phase="train",
        transform=config.train_transforms,
        max_length=2000,
        select_every=1,
    )
    n_training_dataset = dataset.num_classes
    training_label_map = dataset.labels_map

    val_dataset = NumpyDataset(
        phase="val",
        metadata=config.metadata,
        root=config.dataset_directory,
        transform=config.test_transforms,
        img_size=config.img_size,
        max_length=2000,
        select_every=10,
        return_isolation=True,
    )
    validation_label_map = dataset.labels_map

    assert (
        config.num_classes == n_training_dataset
    ), f"The 'user_configs.yaml file has num_classes set to {config.num_classes}, but the training dataset contain {n_training_dataset} distinct classes.'"
    for v_lbl in validation_label_map:
        assert (
            v_lbl in training_label_map
        ), f"The validation label {v_lbl} in not in the training labels: {training_label_map}"

    with open(os.path.join(config.save_directory, "re-identification_metadata.yaml"), "w") as f:
        yaml.dump(
            dict(input_shape=[224, 224], nb_features=config.model_config.n_output_embd, identities=training_label_map),
            f,
        )

    model = PtReIDModel(config.model_config, pretrained=True)
    objective = ArcFaceWithCrossEntropyLoss(
        num_classes=dataset.num_classes, embedding_size=config.model_config.n_output_embd, margin=0.5, scale=64
    )

    params = chain(model.parameters(), objective.arcface_loss.parameters())
    optimizer = SGD(params=params, lr=0.001, momentum=0.9)
    min_lr = optimizer.defaults.get("lr") * 1e-3

    epochs = config.epochs
    warmup_epochs = int(2 * epochs / 3)
    cosine_epochs = epochs - warmup_epochs

    warmup = LambdaLR(optimizer, lr_lambda=lambda epoch: (epoch + 1) / warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=min_lr)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    trainer = ClassificationTrainerWithValidation(
        dataset=dataset,
        val_dataset=val_dataset,
        save_dir=config.save_directory,
        checkpoint_name="precision_track_re-identificator.pth",
        model=model,
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
    ckpt_path = os.path.abspath(os.path.join(config.save_directory, "precision_track_re-identificator.pth"))
    model = PtReIDModel(config=config.model_config, checkpoint=ckpt_path)

    test_metrics(config, model)
    test_classification(config, model)

    print_info("Done testing.")


def deploy(config):
    ckpt_path = os.path.abspath(os.path.join(config.save_directory, "precision_track_re-identificator.pth"))
    model = PtReIDModel(config=config.model_config, checkpoint=ckpt_path)
    model.eval()

    deploy_model(config, model)

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

    labels_path = os.path.abspath(os.path.join(config.dataset_directory, "labels.csv"))
    if os.path.exists(labels_path) and os.path.isfile(labels_path):
        print_info(
            f"Labels file '{labels_path}' found. Will NOT be creating a new one. If you wish to update your labels file, simply delete the existing one."
        )
    else:
        print_info(f"Labels file '{labels_path}' not found. Will be creating a new one.")
        create_labels_file(config.dataset_directory, config.val_split, config.seed)
    config.labels_path = labels_path

    os.makedirs(config.save_directory, exist_ok=True)

    config.img_size = (224, 224)

    config.train_transforms = T.Compose(
        [
            T.ToPILImage(),
            T.RandomResizedCrop(size=config.img_size, scale=(0.8, 1.0)),
            T.RandAugment(num_ops=2, magnitude=20),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    config.test_transforms = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    config.model_config = Dict(
        dict(
            n_embd=768,
            n_output_embd=16,
            n_layers=1,
            n_classes=config.num_classes,
            dropout=0.0,
        )
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
