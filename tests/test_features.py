import timm
from wildlife_tools.data import FeatureDataset
from wildlife_tools.features import DeepFeatures, DataToMemory


def test_features_deep(dataset_deep):
    backbone = timm.create_model('hf-hub:BVRA/MegaDescriptor-T-224', num_classes=0, pretrained=True)
    extractor = DeepFeatures(backbone)
    output = extractor(dataset_deep)
    assert type(output) == FeatureDataset
    assert len(output) == len(dataset_deep)
    assert len(output[0][0]) == 768


def test_data_memory(dataset_deep):
    extractor = DataToMemory()
    output = extractor(dataset_deep)
    assert len(output) == len(dataset_deep)
