# This file contains several definitions of variables and data loaders that will be used for training.
# You may change some values in this file.

from medmnist import INFO
import medmnist
import torch.nn as nn
import torch.utils.data as data
import torchio as tio

# IMPORTANT HYPERPARAMETERS!!!!!! modify carefully
time_of_tests = 1               # choose an integer from 0 to 999 for separating several tests
num_of_models = 15              # number of models to create; a multiple of 3 is recommended (default : 15)

num_epochs = 500                # (default : 500)
batch_size = 32                 # (default : 32)

lr = 0.001                      # (default : 0.001)
weight_decay = 0.0005           # (default : 0.0005)

# medmnist data information
data_flag = 'organmnist3d'      # options : organ, nodule, fracture, adrenal, vessel, synapse
download = True

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])

# transformations and data loaders
transform_train = tio.Compose([tio.ToCanonical(),
                               tio.RandomNoise(p=0.2),
                               tio.RandomFlip(p=0.4),
                               tio.RandomBlur(p=0.2),
                               tio.RandomBiasField(p=0.1),
                               tio.RandomMotion(p=0.2, degrees=10, translation=10, num_transforms=2),
                               tio.RandomAffine(p=0.2, degrees=(2,2)),
                               tio.ZNormalization(p=1),     # DELETE this line for organ and vessel
                               tio.OneHot(num_classes=n_classes)
                               ])
transform_test = tio.Compose([tio.ToCanonical(),
                              tio.ZNormalization(p=1),      # DELETE this line for organ and vessel
                              tio.OneHot(num_classes=n_classes)
                              ])

train_data = DataClass(split='train', transform = transform_train, download = download)
valid_data = DataClass(split='val', transform = transform_test, download = download)
test_data = DataClass(split='test', transform = transform_test, download = download)

train_loader = data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
train_loader_at_eval = data.DataLoader(dataset=valid_data, batch_size=2*batch_size, shuffle=False)
test_loader = data.DataLoader(dataset=test_data, batch_size=2*batch_size, shuffle=False)

# loss functions suggested by medmnist
if task == "multi-label, binary-class":
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()
