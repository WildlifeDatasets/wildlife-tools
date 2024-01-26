import timm
from wildlife_tools.features import SIFTFeatures, DeepFeatures, DataToMemory


def test_features_deep(dataset_deep):
    backbone = timm.create_model('hf-hub:BVRA/MegaDescriptor-T-224', num_classes=0, pretrained=True)
    extractor = DeepFeatures(backbone)
    output = extractor(dataset_deep)
    assert len(output) == len(dataset_deep)
    assert output.shape[1] == 768


def test_features_sift(dataset_sift):
    extractor = SIFTFeatures()
    output = extractor(dataset_sift)
    print(output)
    assert len(output) == len(dataset_sift)
    assert output[0].shape[1] == 128


def test_data_memory(dataset_deep):
    extractor = DataToMemory()
    output = extractor(dataset_deep)
    assert len(output) == len(dataset_deep)
