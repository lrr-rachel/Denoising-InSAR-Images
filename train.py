################# Training the UNet 1E+2D Model with cross-validation, by RLin CS@UoB #################
from __future__ import print_function, division
import argparse
from ctypes import resize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import  transforms
import os
import copy
import glob
from torch.utils.data import Dataset
from skimage import io, transform
# from networks import UnetGenerator
from networks import TwoDecoderUnetGenerator
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import random
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description='Main Training Code')
parser.add_argument('--network', type=str, default='unet', help='select model: unet or dncnn')
parser.add_argument('--root_distorted', type=str, default='DST', help='noisy dataset')
parser.add_argument('--root_restored', type=str, default='D', help='clean dataset')
parser.add_argument('--resultDir', type=str, default='results_final', help='output directory')
parser.add_argument('--unetdepth', type=int, default=5, metavar='N',  help='number of unet depths (default: 5)')
parser.add_argument('--numframes', type=int, default=32, metavar='N',  help='batch number (default: 32)')
parser.add_argument('--maxepoch', type=int, default=200, help='number of epochs to train. default: 200')
parser.add_argument('--savemodel_epoch', type=int, default=20, help='save model every _ epochs. default: 20 (save model every 20 epochs)')
parser.add_argument('--cropsize', type=int, default=0)
parser.add_argument('--savemodelname', type=str, default='model')
parser.add_argument('--NoNorm', action='store_false', help='Run test only')
parser.add_argument('--deform', action='store_true', help='Run test only')
parser.add_argument('--retrain', action='store_true')
parser.add_argument('--topleft', action='store_true', help='crop using top left')
parser.add_argument('--resizedata',default='true', help='resize input')
parser.add_argument('--resize_height',type=int,default=256,help='resize height')
parser.add_argument('--resize_width',type=int,default=256, help='resize width')


args = parser.parse_args()

network = args.network
root_distorted = args.root_distorted
root_restored  = args.root_restored
resultDir = args.resultDir
unetdepth = args.unetdepth
numframes = args.numframes
maxepoch = args.maxepoch
savemodel_epoch = args.savemodel_epoch
cropsize = args.cropsize
savemodelname = args.savemodelname
NoNorm = args.NoNorm
deform = args.deform
retrain = args.retrain
topleft = args.topleft
resizedata = args.resizedata
resize_height = args.resize_height
resize_width = args.resize_width


if not os.path.exists(resultDir):
    os.mkdir(resultDir)

def shiftToMean(data):
    return (np.array(data) - np.mean(data))

class UNetDataset(Dataset):
    def __init__(self, root_distorted, root_restored='', network='unet', numframes=1, transform=None):
        self.root_distorted = root_distorted
        self.root_restored = root_restored
        self.transform = transform
        if len(root_restored)==0:
            self.filesnames = glob.glob(os.path.join(root_distorted,'**_restored/*.png'))
        else:
            self.filesnames = glob.glob(os.path.join(root_distorted,'*.png'))
        self.numframes = numframes
    def __len__(self):
        return len(self.filesnames)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #curf = int(subname[-1][:-4])
        #halfcurf = int(self.numframes/2)
        if len(self.root_restored)==0:
            totalframes = len(glob.glob(os.path.join(os.path.dirname(os.path.abspath(self.filesnames[idx])), '*.png')))
        else:
            totalframes = len(self.filesnames)
        if numframes > 1:
            otherindx = random.sample(range(totalframes),numframes-1)
            rangef = np.unique(np.append(idx, otherindx))
            while len(rangef) < numframes:
                rangef = np.unique(np.append(rangef, random.sample(range(totalframes),numframes-len(rangef))))
        for f in rangef:
            subname = self.filesnames[f].split("\\")
            # read distorted image
            temp = io.imread(os.path.join(self.root_distorted,subname[-1]),as_gray=True)
            # temp = shiftToMean(temp)
            temp = temp.astype('float32')
            temp = temp[..., np.newaxis]

            # read restored image
            tempgt = io.imread(os.path.join(self.root_restored,subname[-1]),as_gray=True)
            # tempgt = shiftToMean(tempgt)
            tempgt = tempgt.astype('float32')
            tempgt = tempgt[..., np.newaxis]
            if f==rangef[0]:
                image = temp/255.
                groundtruth = tempgt/255
                
            else:
                image = np.append(image,temp/255.,axis=2)
                groundtruth = np.append(groundtruth,tempgt/255.,axis=2)
        sample = {'image': image, 'groundtruth': groundtruth}
        if self.transform:
            sample = self.transform(sample)
        return sample


class RandomCrop(object):
    def __init__(self, output_size, topleft=False):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.topleft = topleft
    def __call__(self, sample):
        image, groundtruth = sample['image'], sample['groundtruth']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        if (h > new_h) and (not topleft):
            top = np.random.randint(0, h - new_h)
        else:
            top = 0
        if (w > new_w) and (not topleft):
            left = np.random.randint(0, w - new_w)
        else:
            left = 0
        image = image[top: top + new_h,
                      left: left + new_w]
        groundtruth = groundtruth[top: top + new_h,
                      left: left + new_w]
        return {'image': image, 'groundtruth': groundtruth}

class ToTensor(object):
    def __init__(self, network='unet'):
        self.network = network
    def __call__(self, sample):
        image, groundtruth = sample['image'], sample['groundtruth']
        # swap color axis because
        # numpy image: H x W x B
        # torch image: B x H x W
        image = image.transpose((2, 0, 1))
        groundtruth = groundtruth.transpose((2, 0, 1))
        image = torch.from_numpy(image.copy())
        groundtruth = torch.from_numpy(groundtruth.copy())
        # image
        vallist = [0.5]*image.shape[0]
        normmid = transforms.Normalize(vallist, vallist)
        image = normmid(image)
        # ground truth
        vallist = [0.5]*groundtruth.shape[0]
        normmid = transforms.Normalize(vallist, vallist)
        groundtruth = normmid(groundtruth)
        # torch image: B x 1 x H x W
        image = image.unsqueeze(1)
        groundtruth = groundtruth.unsqueeze(1)
        return {'image': image, 'groundtruth': groundtruth}

class RandomFlip(object):
    def __call__(self, sample):
        image, groundtruth = sample['image'], sample['groundtruth']
        op = np.random.randint(0, 3)
        if op<2:
            image = np.flip(image,op)
            groundtruth = np.flip(groundtruth,op)
        return {'image': image, 'groundtruth': groundtruth}

def readimage(filename):
    # read distorted image
    temp = io.imread(filename,as_gray=True)
    temp = temp.astype('float32')
    temp = temp[1: 225, 1: 225]
    image = temp/255.
    image = np.expand_dims(image, axis=2)
    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image)
    vallist = [0.5]*image.shape[0]
    normmid = transforms.Normalize(vallist, vallist)
    image = normmid(image)
    image = image.unsqueeze(0)
    return image

def resizeimage(path, width, height):
    print("[INFO] resizing inputs to ", resize_width, 'x', resize_height)
    dirs_dis = os.listdir(path)
    for item in dirs_dis:
        if os.path.isfile(path+'/'+item):
            im = Image.open(path+'/'+item)
            f, e = os.path.splitext(path+'/'+item)
            imResize = im.resize((width,height), Image.ANTIALIAS)
            imResize.save(f + '.png', 'PNG', quality=90)
            # print('image resize')

# =====================================================================

# resize input image size
if resizedata == 'true':
    print('[INFO] resize noisy input...')
    resizeimage(root_distorted,resize_width,resize_height)
    print('[INFO] resize clean input...')
    resizeimage(root_restored,resize_width,resize_height)

# data loader
print("[INFO] Loading Data")
if network == 'unet':
    if cropsize==0:
        unetdataset = UNetDataset(root_distorted=root_distorted,
                                        root_restored=root_restored, network=network, numframes=numframes,
                                        transform=transforms.Compose([RandomFlip(),ToTensor(network=network)]))
    else:
        unetdataset = UNetDataset(root_distorted=root_distorted,
                                        root_restored=root_restored, network=network, numframes=numframes,
                                        transform=transforms.Compose([RandomCrop(cropsize, topleft=topleft),RandomFlip(),ToTensor(network=network)]))
                                        

#     if retrain:
#         model.load_state_dict(torch.load(os.path.join(resultDir,'best_'+ savemodelname+'.pth.tar'),map_location=device))


# 5-fold cross validation
'''CV'''
num_folds = 5
num_epochs=maxepoch
kf = KFold(n_splits=num_folds, shuffle=True, random_state=1)
# Initialize lists to store training and validation losses for each fold
all_train_losses = []
all_val_losses = []

dataset_list = list(unetdataset)

for fold, (train_indices, val_indices) in enumerate(kf.split(dataset_list)):
    print(f'Fold {fold + 1}/{num_folds}')

    # Create lists to hold training and validation samples
    train_data = [dataset_list[i] for i in train_indices]
    val_data = [dataset_list[i] for i in val_indices]

    # Create a new model for each fold
    model = TwoDecoderUnetGenerator(input_nc=1, output_nc=1, num_downs=unetdepth, deformable=deform, norm_layer=NoNorm)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Print the chosen device
    print(f"Using device: {device}")

    if torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs!')
        model = nn.DataParallel(model)

    model = model.to(device)
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.000001)

    best_val_loss = float('inf')
    #  Initialize lists to store training and validation losses for this fold
    train_losses = []
    val_losses = []

    # Train the model for each epoch
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i in random.sample(range(len(train_data)), len(train_data)):
            sample = train_data[i]
            inputs, labels = sample['image'].to(device), sample['groundtruth'].to(device)

            optimizer.zero_grad()

            groundtruth_outputs, noise_outputs = model(inputs)
            groundtruth_loss = criterion(groundtruth_outputs, labels)
            noise_loss = criterion(noise_outputs, inputs - labels)
            loss = groundtruth_loss + noise_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_data)
        print(f'Fold {fold + 1}, Epoch {epoch + 1}, Train Loss: {epoch_loss:.4f}')

        # Validate the model for each epoch
        model.eval()
        validate_loss = 0.0

        for i in range(len(val_data)):
            sample = val_data[i]
            inputs, labels = sample['image'].to(device), sample['groundtruth'].to(device)

            groundtruth_outputs, noise_outputs = model(inputs)
            groundtruth_loss = criterion(groundtruth_outputs, labels)
            noise_loss = criterion(noise_outputs, inputs - labels)
            loss = groundtruth_loss + noise_loss

            validate_loss += loss.item()

        epoch_val_loss = validate_loss / len(val_data)
        print(f'Fold {fold + 1}, Epoch {epoch + 1}, Validation Loss: {epoch_val_loss:.4f}')

        train_losses.append(epoch_loss)
        val_losses.append(epoch_val_loss)

        # Save the training and validation losses for this fold after each epoch
        np.save(f'data_array_train_fold{fold}_epoch{epoch}.npy', np.array(train_losses))
        np.save(f'data_array_val_fold{fold}_epoch{epoch}.npy', np.array(val_losses))

        if (epoch % savemodel_epoch) == 0:
            torch.save(model.state_dict(), os.path.join(resultDir, savemodelname + '_ep'+str(epoch)+'.pth.tar'))

        # save the best model for each fold
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model = copy.deepcopy(model.state_dict())
            torch.save(best_model, os.path.join(resultDir,'best_model.pth.tar'))


    # Append the loss lists for this fold to the overall lists
    all_train_losses.append(train_losses)
    all_val_losses.append(val_losses)

# Calculate and print the average training and validation losses across all folds
avg_train_loss = np.mean(all_train_losses, axis=0)
avg_val_loss = np.mean(all_val_losses, axis=0)
print(f'Average Training Loss (across all folds and epochs): {avg_train_loss}')
print(f'Average Validation Loss (across all folds and epochs): {avg_val_loss}')

np.save(f'average_train.npy', np.array(avg_train_loss))
np.save(f'average_val.npy', np.array(avg_val_loss))

#also save the best model
torch.save(best_model, 'best_model.pth.tar')


