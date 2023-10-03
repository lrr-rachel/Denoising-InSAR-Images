# Implementation of the 1E2D U-Net for InSAR Image Denoising

This is a PyTorch implementation of a 1E+2D U-Net designed to mitigate the noisy effects on InSAR images. 

This also includes synthetic data generation code that has been adapted and modified from [N. Anantrasirichai, J. Biggs, F. Albino, D. Bull, A deep learning approach to detecting volcano deformation from satellite imagery using synthetic datasets, Remote Sensing of Environment, 230, 2019](https://github.com/pui-nantheera/Synthetic_InSAR_image/tree/main). 

The originasl U-Net architecture is based on the work by [O. Ronneberger, P. Fischer, and T. Brox, 'U-Net: Convolutional Networks for Biomedical Image Segmentation,' arXiv.org, 18-May-2015](https://arxiv.org/abs/1505.04597). Modifications have been made to create a revised 1E+2D U-Net architecture.

## 1E+2D U-Net Model
* One-encoder, two-decoder architecture
* Decoder 1 reconstructs Groundtruth/Deformation D
* Decoder 2 reconstructs Noise ST

## Dependencies
The code has been mainly developed and tested in:
pytorch 2.0.1 cuda 11.8 (latest released version)

Select one of the followings for setting up training environment:
* [Pytorch](https://pytorch.org/) 
* BlueCrystal4: Recommand building your own anaconda environment. Please see detailed instrusctions on [installing your own conda env on bc4](https://www.acrc.bris.ac.uk/protected/hpc-docs/software/python_conda.html)

For Synthetic Dataset Preprocessing and Generation: MatlabR2023b

* The Matlab code generates synthetic deformations (D) and combines them with noise (ST) to create a dataset consisting of clean-noisy pairs. 





## Training
* Specify the noisy data directory, clean data directory, output directory, etc.
* ```train.py --help``` to see useful help messages.
* Demo: 
```
python train.py --network unet --root_distorted DST_png --root_restored D_png --resultDir unetresults --maxepoch 200 
```
* The best-trained model will be saved in the output directory specified.

Note: Please review the help messages for command line arguments before proceeding.


## Testing or Loading the Pretrained Model
* Run ```denoise.py```
* ```denoise.py --help``` to see useful help messages.
* Specify the test data directory, output result directory, etc.
* Demo: 
```
python denoise.py --network unet --root_distorted ./data/Test_png/ --resultDir unetresults
```
* ```denoise.py``` loads and tests the pretrained model using noisy test dataset, then outputs corresponding (in unwrapped format): 
    * Deformation (denoised output)
	* Noise (noise removed)

Note:
* Please refer to the help messages for command line arguments before continuing.
* The resultDir must be the output directory created in the Training.
* The resultDir must contain the trained model.
* The test data (e.g., Test_png) must first be 
	* normalized (see ```Norm.m```);
	* AND at the same size as training data (e.g., an image size of 256x256 pixels if using the pretrained model stored in 'results_final').

## Wrapping
See example code ```Wrap_denorm.m```. When converting the image to a wrapped format, it is recommanded to de-normalize it first using its scaling values in the normalization process.