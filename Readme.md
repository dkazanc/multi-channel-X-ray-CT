# Joint image reconstruction method with correlative multi-channel prior for X-ray spectral computed tomography (SOFTWARE)

**The presented software reproduce the numerical results of Inverse Problem [paper](https://doi.org/10.1088/1361-6420/aaba86). Software generate realisitc 
sythetic multi-energetic tomographic projection data and reconstruct it. Novel correlative multi-channel algorithm compared with other state-of-the-art 
reconstruction algorithms. Note that the GPU Nvidia card is required to use the software.** 

## General software prerequisites
 * [MATLAB](http://www.mathworks.com/products/matlab/) 
 * C compilers and nvcc (CUDA SDK) compilers
 * [ASTRA-toolbox](https://www.astra-toolbox.com/) 
 
## Software dependencies for data generation script: 
 * [Photon Attenuation](https://uk.mathworks.com/matlabcentral/fileexchange/12092-photonattenuation)
 * [Spektr](http://istar.jhu.edu/downloads/) 

## Software dependencies for data reconstruction script: 
 * [CCPi-RegularisationToolkit](https://github.com/vais-ral/CCPi-Regularisation-Toolkit)
 * [Spot operator package](http://www.cs.ubc.ca/labs/scl/spot/)

## Installation:
 * See INSTALLATION file for detailed recommendations
 
## Package contents:
 * SpectralData_generation_demo.m - script to generate data
 * SpectralData_noCrime.mat - pre-generated data using the script above
 * SpectralReconDemo.m - script to perform reconstruction of generated data
 * FISTA_REC_rct.m - main reconstruction function

### References:
 * Kazantsev D., Jørgensen J.S., Andersen M., Lionheart W.R., Lee P.D. and Withers P.J., 2018. Joint image reconstruction method with correlative multi-channel prior for X-ray spectral computed tomography. Inverse Problems. 
 
### License:
GNU GENERAL PUBLIC LICENSE v.3

### Questions/Comments
can be addressed to Daniil Kazantsev at dkazanc@hotmail.com
