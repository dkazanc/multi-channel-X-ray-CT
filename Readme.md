# Joint image reconstruction method with correlative multi-channel prior for X-ray spectral computed tomography (SOFTWARE)

**The presented software reproduces numerical results of Inverse Problems journal [paper](https://doi.org/10.1088/1361-6420/aaba86). Software generates realisitic synthetic multi-energetic tomographic projection data and reconstructs it. Novel prior-correlative multi-channel algorithm is compared with other state-of-the-art reconstruction algorithms. Note that a GPU (Nvidia) card is required to use the software.** 

## General software prerequisites
 * [MATLAB](http://www.mathworks.com/products/matlab/) 
 * C compilers and nvcc [CUDA SDK](https://developer.nvidia.com/cuda-downloads) compilers
 * [ASTRA-toolbox](https://www.astra-toolbox.com/) 
 
## Software dependencies for data generation script: 
 * [Photon Attenuation](https://uk.mathworks.com/matlabcentral/fileexchange/12092-photonattenuation)
 * [Spektr](http://istar.jhu.edu/downloads/) 

## Software dependencies for data reconstruction script: 
 * [CCPi-RegularisationToolkit](https://github.com/vais-ral/CCPi-Regularisation-Toolkit)
 * [Spot operator package](http://www.cs.ubc.ca/labs/scl/spot/)

## Installation and known Issues:
 * See INSTALLATION file for detailed recommendations
 * There is an [issue](https://github.com/dkazanc/multi-channel-X-ray-CT/issues/1) related to Photon Attenuation software
 
## Package contents:
 * SpectralData_generation_demo.m - script to generate data
 * SpectralData_noCrime.mat - pre-generated data obtained with the script above
 * SpectralReconDemo.m - script to perform reconstruction of generated data
 * FISTA_REC_rct.m - main reconstruction function (FISTA algorithm)

### References:
 * [Kazantsev D., Jørgensen J.S., Andersen M., Lionheart W.R., Lee P.D. and Withers P.J., 2018. Joint image reconstruction method with correlative multi-channel prior for X-ray spectral computed tomography. Inverse Problems](https://doi.org/10.1088/1361-6420/aaba86)
 * [SIAM 2018 presentation](https://github.com/dkazanc/multi-channel-X-ray-CT/blob/master/docs/SIAM_Kazantsev18.pdf)
 
### License:
GNU GENERAL PUBLIC LICENSE v.3

### Questions/Comments
can be addressed to Daniil Kazantsev at dkazanc@hotmail.com
