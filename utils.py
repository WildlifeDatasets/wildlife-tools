import os
from datetime import datetime

def setup_folder(config_filename, root='runs', name=''):
    name, _ = os.path.splitext(os.path.basename(config_filename))
    timestamp = datetime.now().strftime("%b%d-%H-%M-%S-%f")[:-2]
    folder = os.path.join(root, name + '_' + timestamp)
    os.makedirs(folder, exist_ok=True)
    return folder