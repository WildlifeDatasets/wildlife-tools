
from wildlife_tools.data.prepare import prepare_functions

requests = [
    {'name': 'NyalaData', 'root': '/mnt/data/turtles/datasets/datasets/NyalaData'},
    {'name': 'ZindiTurtleRecall', 'root': '/mnt/data/turtles/datasets/datasets/ZindiTurtleRecall'},
    {'name': 'BelugaID', 'root': '/mnt/data/turtles/datasets/datasets/BelugaID'},
    {'name': 'BirdIndividualID', 'root': '/mnt/data/turtles/datasets/datasets/BirdIndividualID'},
    {'name': 'SealID', 'root': '/mnt/data/turtles/datasets/datasets/SealID'},
    {'name': 'FriesianCattle2015', 'root': '/mnt/data/turtles/datasets/datasets/FriesianCattle2015'},
    {'name': 'ATRW', 'root': '/mnt/data/turtles/datasets/datasets/ATRW'},
    {'name': 'NDD20', 'root': '/mnt/data/turtles/datasets/datasets/NDD20'},
    {'name': 'SMALST', 'root': '/mnt/data/turtles/datasets/datasets/SMALST'},
    {'name': 'SeaTurtleIDHeads', 'root': '/mnt/data/turtles/datasets/datasets/SeaTurtleIDHeads'},
    {'name': 'AAUZebraFishID', 'root': '/mnt/data/turtles/datasets/datasets/AAUZebraFishID'},
    {'name': 'CZoo', 'root': '/mnt/data/turtles/datasets/datasets/CZoo'},
    {'name': 'CTai', 'root': '/mnt/data/turtles/datasets/datasets/CTai'},
    {'name': 'Giraffes', 'root': '/mnt/data/turtles/datasets/datasets/Giraffes'},
    {'name': 'HyenaID2022', 'root': '/mnt/data/turtles/datasets/datasets/HyenaID2022'},
    {'name': 'MacaqueFaces', 'root': '/mnt/data/turtles/datasets/datasets/MacaqueFaces'},
    {'name': 'OpenCows2020', 'root': '/mnt/data/turtles/datasets/datasets/OpenCows2020'},
    {'name': 'StripeSpotter', 'root': '/mnt/data/turtles/datasets/datasets/StripeSpotter'},
    {'name': 'AerialCattle2017', 'root': '/mnt/data/turtles/datasets/datasets/AerialCattle2017'},
    {'name': 'GiraffeZebraID', 'root': '/mnt/data/turtles/datasets/datasets/GiraffeZebraID'},
    {'name': 'IPanda50', 'root': '/mnt/data/turtles/datasets/datasets/IPanda50'},
    {'name': 'WhaleSharkID', 'root': '/mnt/data/turtles/datasets/datasets/WhaleSharkID'},
    {'name': 'FriesianCattle2017', 'root': '/mnt/data/turtles/datasets/datasets/FriesianCattle2017'},
    {'name': 'Cows2021', 'root': '/mnt/data/turtles/datasets/datasets/Cows2021'},
    {'name': 'LeopardID2022', 'root': '/mnt/data/turtles/datasets/datasets/LeopardID2022'},
    {'name': 'NOAARightWhale', 'root': '/mnt/data/turtles/datasets/datasets/NOAARightWhale'},
    {'name': 'WNIGiraffes', 'root': '/mnt/data/turtles/datasets/datasets/WNIGiraffes'},
    {'name': 'HappyWhale', 'root': '/mnt/data/turtles/datasets/datasets/HappyWhale'},
    {'name': 'HumpbackWhaleID', 'root': '/mnt/data/turtles/datasets/datasets/HumpbackWhaleID'}, 
    {'name': 'LionData', 'root': '/mnt/data/turtles/datasets/datasets/LionData'},
]

if __name__ == '__main__':
    for config in configs:
        name = config.pop('name')
        print(f'Processing: {name}')
        processing_function[name](**config)