################# Testing or Loading the pretrained model, by RLin CS@UoB #################
import argparse
import torch
import os
import glob
import numpy as np
import torch
import torch.nn as nn
from skimage import io, transform
from torchvision import transforms
from networks import *
import cv2
from torch.autograd import Variable
from models import DnCNN
import torchvision.transforms as T
import cv2
import matplotlib.pyplot as plt
from skimage import img_as_uint
from PIL import Image

parser = argparse.ArgumentParser(description="Test")
parser.add_argument('--network', type=str, default='unet',help='select model: unet or dncnn')
parser.add_argument('--savemodelname', type=str, default='model_gt', help='save output gt images')
parser.add_argument('--savemodelname_noise', type=str, default='model_noise')
parser.add_argument('--root_distorted', type=str, default='Test_png/', help='test dataset')

parser.add_argument('--resultDir', type=str, default='results_final', help='save output images. default: results (same dir as resultDir in train.py)')
parser.add_argument('--deform', action='store_true', help='Run test only')
parser.add_argument('--NoNorm', action='store_false', help='Run test only')
args = parser.parse_args()

network = args.network
savemodelname = args.savemodelname
savemodelname_noise = args.savemodelname_noise
root_distorted = args.root_distorted
resultDir = args.resultDir
deform = args.deform
NoNorm = args.NoNorm

def normalize(data):
    return data/255.

def readimage(filename):
    # read distorted image
    temp = io.imread(filename,as_gray=True)
    temp = temp.astype('float32')
    image = temp/255.
    # image = image[1: 225,1: 225]
    image = np.expand_dims(image, axis=2)
    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image)
    vallist = [0.5]*image.shape[0]
    normmid = transforms.Normalize(vallist, vallist)
    image = normmid(image)
    image = image.unsqueeze(0)
    return image

def im2double(im):
  np.seterr(divide='ignore', invalid='ignore')
  min_val = np.min(im.ravel())
  max_val = np.max(im.ravel())
  out = (im.astype('float') - min_val) / (max_val - min_val)
  return out

# =======TESTING==============================================================

unetdepth = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

resultDirOutImg = os.path.join(resultDir,savemodelname)
if not os.path.exists(resultDirOutImg):
    os.mkdir(resultDirOutImg)

resultDirOutImg_noise = os.path.join(resultDir,savemodelname_noise)
if not os.path.exists(resultDirOutImg_noise):
    os.mkdir(resultDirOutImg_noise)

unwrappedDir = os.path.join(resultDirOutImg,'unwrapped')
if not os.path.exists(unwrappedDir):
    os.mkdir(unwrappedDir)

unwrappedDir_noise = os.path.join(resultDirOutImg_noise,'unwrapped')
if not os.path.exists(unwrappedDir_noise):
    os.mkdir(unwrappedDir_noise)


if network == 'unet':
    print("loading UNet")

    model = TwoDecoderUnetGenerator(input_nc=1, output_nc=1, num_downs=unetdepth, deformable=deform, norm_layer=NoNorm)

    # load saved model in restultDir
    checkpoint = torch.load(os.path.join(resultDir,'best_model.pth.tar'), map_location=device)

    '''remove module (if trained with multi-GPUs)'''
    # print("key:",checkpoint.keys())
    for key in list(checkpoint.keys()):
        if 'module.' in key:
            checkpoint[key[7:]] = checkpoint[key] #delete term "module"
            del checkpoint[key]
    # print("key2:",checkpoint.keys())

    model.load_state_dict(checkpoint)
    model = model.eval()
    model = model.to(device)
    
    # print weights
    # for name,param in model.named_parameters():
    # 	if param.requires_grad:
    # 		print(name,param.data)
    		
# =====================================================================

# save unwrapped output images
filesnames = glob.glob(os.path.join(root_distorted,'*.png'))
if network == 'unet':
    for i in range(len(filesnames)):
        curfile = filesnames[i]
        inputs = readimage(curfile)
        subname = curfile.split("\\")
        print(subname)
        inputs = inputs.to(device)

        with torch.no_grad():
            groundtruth_outputs, noise_outputs = model(inputs)

            groundtruth_outputs = groundtruth_outputs.squeeze(0)
            groundtruth_outputs = groundtruth_outputs.cpu().numpy() 
            groundtruth_outputs = groundtruth_outputs.transpose((1, 2, 0))
            groundtruth_outputs = (groundtruth_outputs*0.5 + 0.5)*255

            noise_outputs = noise_outputs.squeeze(0)
            noise_outputs = noise_outputs.cpu().numpy() 
            noise_outputs = noise_outputs.transpose((1, 2, 0))
            noise_outputs = (noise_outputs*0.5 + 0.5)*255


            # output groundtruth
            imfile = Image.fromarray(groundtruth_outputs[:, :, 0].astype('uint8'), 'L')

            # output noise
            imfile_noise = Image.fromarray(noise_outputs[:, :, 0].astype('uint8'), 'L')

            imfile.save(os.path.join(unwrappedDir, subname[-1]))
            imfile_noise.save(os.path.join(unwrappedDir_noise, subname[-1]))

