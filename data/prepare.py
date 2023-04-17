'''
Descibed recommended processing of datasets (apply bbox/segmentations, resize, etc.)
Saves all processes and resized images to new folders
'''
import os
import sys
import torchvision.transforms as T
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image
# TODO: fix imports

# Wildlife training
root = os.path.join('..')
sys.path.append(root)
from wildlife_tools.data import ImageDataset
# Wildlife datasets
root = os.path.join('../../datasets')
sys.path.append(root)
from wildlife_datasets import datasets


def resize_dataset(dataset_factory, new_root, size=256, img_load='bbox'):
    '''
    Example usage:

    from utils import resize_turtles

    dataset_factory = datasets.SeaTurtleID('datasets/SeaTurtleID')
    resize_turtles(dataset_factory, 'data/256x256_bbox', max_size=256, img_load='bbox')
    '''

    dataset = ImageDataset(
        dataset_factory.df,
        dataset_factory.root,
        transform=T.Resize(size=size),
        img_load=img_load)

    for i in tqdm(range(len(dataset))):
        path = os.path.join(new_root, dataset.df.iloc[i]['path'])
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        image, _ = dataset[i]
        image.save(path)

def save_dataframe(dataset_factory, new_root):
    df_simplified = dataset_factory.df[['id', 'identity', 'path']]
    assert type(df_simplified.index) == pd.RangeIndex
    df_simplified.to_csv(os.path.join(new_root, 'annotations.csv'))

def prepare_sea_turtle_id_heads(root, new_root='data/SeaTurtleIDHeads', size=256):
    dataset_factory = datasets.SeaTurtleIDHeads(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load='full')
    df = dataset_factory.df[['id', 'identity', 'path', 'date']]
    df.to_csv(os.path.join(new_root, 'annotations.csv'))

def prepare_zebra_fish_id(root, new_root='data/AAUZebraFishID', size=256):
    dataset_factory = datasets.AAUZebraFishID(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load='bbox')
    save_dataframe(dataset_factory, new_root)

def prepare_czoo(root, new_root='data/CZoo', size=256):
    dataset_factory = datasets.CZoo(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load='full')
    save_dataframe(dataset_factory, new_root)

def prepare_ctai(root, new_root='data/CTai', size=256):
    dataset_factory = datasets.CTai(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load='full')
    save_dataframe(dataset_factory, new_root)

def prepare_giraffes(root, new_root='data/Giraffes', size=256):
    dataset_factory = datasets.Giraffes(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load='full')
    save_dataframe(dataset_factory, new_root)

def prepare_hyena_id_2022(root, new_root='data/HyenaID2022', size=256):
    dataset_factory = datasets.HyenaID2022(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load='bbox')
    save_dataframe(dataset_factory, new_root)

def prepare_macaque_faces(root, new_root='data/MacaqueFaces', size=256):
    dataset_factory = datasets.MacaqueFaces(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load='full')
    df = dataset_factory.df[['id', 'identity', 'path', 'date']]
    df.to_csv(os.path.join(new_root, 'annotations.csv'))

def prepare_open_cows_2020(root, new_root='data/OpenCows2020', size=256):
    dataset_factory = datasets.OpenCows2020(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load='full')
    save_dataframe(dataset_factory, new_root)

def prepare_stripe_spotter(root, new_root='data/StripeSpotter', size=256):
    dataset_factory = datasets.StripeSpotter(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load='bbox')
    save_dataframe(dataset_factory, new_root)

def prepare_aerial_cattle_2017(root, new_root='data/AerialCattle2017', size=256):
    dataset_factory = datasets.AerialCattle2017(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load='full')
    save_dataframe(dataset_factory, new_root)

def prepare_giraffe_zebra_id(root, new_root='data/GiraffeZebraID', size=256):
    dataset_factory = datasets.GiraffeZebraID(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load='bbox')
    df = dataset_factory.df[['id', 'identity', 'path', 'date']]
    df.to_csv(os.path.join(new_root, 'annotations.csv'))

def prepare_ipanda_50(root, new_root='data/IPanda50', size=256):
    dataset_factory = datasets.IPanda50(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load='full')
    save_dataframe(dataset_factory, new_root)

def prepare_whaleshark_id(root, new_root='data/WhaleSharkID', size=256):
    dataset_factory = datasets.WhaleSharkID(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load='bbox')
    save_dataframe(dataset_factory, new_root)

def prepare_friesian_cattle_2017(root, new_root='data/FriesianCattle2017', size=256):
    dataset_factory = datasets.FriesianCattle2017(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load='full')
    save_dataframe(dataset_factory, new_root)

def prepare_cows2021(root, new_root='data/Cows2021', size=256):
    dataset_factory = datasets.Cows2021(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load='full')
    save_dataframe(dataset_factory, new_root)

def prepare_leopard_id_2022(root, new_root='data/LeopardID2022', size=256):
    dataset_factory = datasets.LeopardID2022(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load='bbox')
    save_dataframe(dataset_factory, new_root)

def prepare_noaa_right_whale(root, new_root='data/NOAARightWhale', size=256):
    dataset_factory = datasets.NOAARightWhale(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load='full')
    save_dataframe(dataset_factory, new_root)

def prepare_wni_giraffes(root, new_root='data/WNIGiraffes', size=256):
    dataset_factory = datasets.WNIGiraffes(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load='full')
    save_dataframe(dataset_factory, new_root)

def prepare_happy_whale(root, new_root='data/HappyWhale', size=256):
    dataset_factory = datasets.HappyWhale(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load='full')
    save_dataframe(dataset_factory, new_root)

def prepare_humpback_whale_id(root, new_root='data/HumpbackWhaleID', size=256):
    dataset_factory = datasets.HumpbackWhaleID(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load='full')
    save_dataframe(dataset_factory, new_root)

def prepare_lion_data(root, new_root='data/LionData', size=256):
    dataset_factory = datasets.LionData(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load='full')
    save_dataframe(dataset_factory, new_root)

def prepare_nyala_data(root, new_root='data/NyalaData', size=256):
    dataset_factory = datasets.NyalaData(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load='full')
    save_dataframe(dataset_factory, new_root)

def prepare_zindi_turtle_recall(root, new_root='data/ZindiTurtleRecall', size=256):
    dataset_factory = datasets.ZindiTurtleRecall(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load='full')
    save_dataframe(dataset_factory, new_root)

def prepare_beluga_id(root, new_root='data/BelugaID', size=256):
    dataset_factory = datasets.BelugaID(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load='bbox')
    df = dataset_factory.df[['id', 'identity', 'path', 'date']]
    df.to_csv(os.path.join(new_root, 'annotations.csv'))

def prepare_bird_individual_id(root, new_root='data/BirdIndividualID', size=256):
    dataset_factory = datasets.BirdIndividualIDSegmented(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load='crop_black')
    save_dataframe(dataset_factory, new_root)

def prepare_seal_id(root, new_root='data/SealID', size=256):
    dataset_factory = datasets.SealIDSegmented(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load='crop_black')
    save_dataframe(dataset_factory, new_root)

def prepare_friesian_cattle_2015(root, new_root='data/FriesianCattle2015', size=256):
    dataset_factory = datasets.FriesianCattle2015(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load='crop_black')
    save_dataframe(dataset_factory, new_root)

def prepare_atrw(root, new_root='data/ATRW', size=256):
    dataset_factory = datasets.ATRW(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load='full')
    save_dataframe(dataset_factory, new_root)

def prepare_ndd20(root, new_root='data/NDD20', size=256):
    dataset_factory = datasets.NDD20(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load='full')
    save_dataframe(dataset_factory, new_root)

def prepare_smalst(root, new_root='data/SMALST', size=256):
    dataset_factory = datasets.SMALST(root)
    dataset = ImageDataset(
        dataset_factory.df,
        dataset_factory.root,
        img_load='full',
    )
    dataset_masks = ImageDataset(
        dataset_factory.df,
        dataset_factory.root,
        img_load='full',
        col_path = 'segmentation'
    )
    for i in tqdm(range(len(dataset))):
        path = os.path.join(new_root, dataset.df.iloc[i]['path'])
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        # Apply mask
        img, _ = dataset[i]
        mask, _ = dataset_masks[i]
        img = Image.fromarray(img * np.array(mask).astype(bool))

        # Crop black parts and resize
        y_nonzero, x_nonzero, _ = np.nonzero(img)
        img = img.crop((np.min(x_nonzero), np.min(y_nonzero), np.max(x_nonzero), np.max(y_nonzero)))
        img = T.Resize(size=size)(img)

        # Save image
        img.save(path)
    save_dataframe(dataset_factory, new_root)


prepare_functions = {
    'NyalaData': prepare_nyala_data,
    'ZindiTurtleRecall': prepare_zindi_turtle_recall,
    'BelugaID': prepare_beluga_id,
    'BirdIndividualID': prepare_bird_individual_id,
    'SealID': prepare_seal_id,
    'FriesianCattle2015': prepare_friesian_cattle_2015,
    'ATRW': prepare_atrw,
    'NDD20': prepare_ndd20,
    'SMALST': prepare_smalst,
    'SeaTurtleIDHeads': prepare_sea_turtle_id_heads,
    'AAUZebraFishID': prepare_zebra_fish_id,
    'CZoo': prepare_czoo,
    'CTai': prepare_ctai,
    'Giraffes': prepare_giraffes,
    'HyenaID2022': prepare_hyena_id_2022,
    'MacaqueFaces': prepare_macaque_faces,
    'OpenCows2020': prepare_open_cows_2020,
    'StripeSpotter': prepare_stripe_spotter,
    'AerialCattle2017': prepare_aerial_cattle_2017,
    'GiraffeZebraID': prepare_giraffe_zebra_id,
    'IPanda50': prepare_ipanda_50,
    'WhaleSharkID': prepare_whaleshark_id,
    'FriesianCattle2017': prepare_friesian_cattle_2017,
    'Cows2021': prepare_cows2021,
    'LeopardID2022': prepare_leopard_id_2022,
    'NOAARightWhale': prepare_noaa_right_whale,
    'WNIGiraffes': prepare_wni_giraffes,
    'HappyWhale': prepare_happy_whale,
    'HumpbackWhaleID': prepare_humpback_whale_id,
    'LionData': prepare_lion_data
}

