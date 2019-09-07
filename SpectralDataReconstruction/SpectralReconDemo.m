% Spectral CT data reconstruction script using ASTRA toolbox
% depends on ASTRA and SPOT Operator
% <<<<<<< ! This code requires GPU/CUDA to run ! >>>>>>>

% This code reproduces numerical results of the paper: 
% Kazantsev D. et al. 2018 
% "Joint image reconstruction method with correlative multi-channel prior for X-ray spectral computed tomography"
% Inverse Problems
% https://doi.org/10.1088/1361-6420/aaba86

close all;clc;clear;
% load generated data
load(sprintf(['..' filesep 'SpectralDataGeneration' filesep 'SpectralData_noCrime.mat'], 1i)); 
% adding paths
addpath(sprintf(['..' filesep 'SupplementaryPackages' filesep], 1i));
addpath(sprintf(['..' filesep 'SupplementaryPackages' filesep 'CCPi-RegularisationToolkit' filesep 'src' filesep 'Matlab' filesep 'mex_compile' filesep 'installed' filesep], 1i));
addpath(sprintf(['..' filesep 'SupplementaryPackages' filesep 'PhotonAttenuation' filesep], 1i));
addpath(sprintf(['..' filesep 'SupplementaryPackages' filesep 'spot' filesep], 1i));
addpath(sprintf(['..' filesep 'SupplementaryPackages' filesep 'gendist' filesep], 1i));

% Set the geometry
B = -log(bsxfun(@times, Y+(Y==0), 1./sb'));
[Ut,Vl] = geocore_phantom(n,Em,'phantom');
Ut_vec = [reshape(Ut(:,:,1),n^2,1), reshape(Ut(:,:,2),n^2,1), reshape(Ut(:,:,3),n^2,1), reshape(Ut(:,:,4),n^2,1)];
Phantom_nbins = Ut_vec*Vl';

theta = (0:p-1)*360/p;   % projection angles
dom_width   = 1.0;       % width of domain in cm
src_to_rotc = 3.0;       % dist. from source to rotation center
src_to_det  = 5.0;       % dist. from source to detector
det_width   = 2.0;       % detector width
vol_geom  = astra_create_vol_geom(n,n);
% Projection geometry
proj_geom = astra_create_proj_geom('fanflat', n*det_width/nd, nd, (pi/180)*theta,...
    n*src_to_rotc/dom_width, n*(src_to_det-src_to_rotc)/dom_width);

% Create the Spot operator for ASTRA using the GPU.
A = (dom_width/n)*opTomo('cuda', proj_geom, vol_geom);

% selecting the number of channels to reconstruct (energy window)
start_channel = 1; end_channel = size(Y,2);
sino = B(:,start_channel:end_channel); 
num_channels = size(sino,2);

% weights for the PWLS model
W = Y/max(Y(:)); 
% Data normalisation 
Yt = Y + (Y==0);
snr_ray = max(1e-5,-log(bsxfun(@rdivide,Yt,sb'))).*sqrt(Yt);
% Obtaing PDF using the geometric mean of a signal
snr_geomean = exp(mean(log(snr_ray)));
snr_geomean = snr_geomean.^2;
snr_geomean = snr_geomean./sum(snr_geomean(:));
%%
fprintf('%s \n', 'Reconstruction using FBP...');
rec_id = astra_mex_data2d('create', '-vol', vol_geom);
% Set up the parameters for a reconstruction algorithm using the GPU
cfg = astra_struct('FBP_CUDA');
cfg.ReconstructionDataId = rec_id;
cfg.FilterType = 'hamming';
cfg.FilterD = 0.8;

X_FBP = zeros(n^2, num_channels);
% loop through each energy channel (use the previous recon as a prior)
for kk=1:num_channels    
    proj_data = reshape(sino(:,kk), p, nd);
    
    % Create projection data object
    proj_id = astra_mex_data2d('create', '-sino', proj_geom, proj_data);
    cfg.ProjectionDataId = proj_id;
    
    % Create the algorithm object from the configuration structure
    alg_id = astra_mex_algorithm('create', cfg);
    
    % Run algorithm
    astra_mex_algorithm('iterate', alg_id, 1);
    
    % Get the result
    rec = astra_mex_data2d('get', rec_id);    
    X_FBP(:,kk) = reshape(rec, n^2, 1)*1.0767e+04;
    
    % Clean up. Note that GPU memory is tied up in the algorithm object,
    % and main RAM in the data objects.
    astra_mex_algorithm('delete', alg_id);
    astra_mex_data2d('delete', proj_id);   
end  
    astra_mex_data2d('delete', rec_id);
    figure; imshow(reshape(X_FBP(:,5), n, n), [0 7]);
%%
fprintf('%s \n', 'PWLS-TV reconstruction using FISTA');
% parameters:
lambdaTV = 4.2e-04;
ROI = find((Phantom_nbins(:,1) >= 0));
clear params
params.Amatrix = A; % projection matrix (opTomo operator)
params.sino = sino; % sinogram (vectorized)
params.iterFISTA = 350; % number of outer iterations (FISTA)
params.REG_method = 'TV'; % regularisation method
params.REG_parameter = lambdaTV; % regularisation parameter
params.REG_iteration = 50;  % number of inner (proximal) iterations
params.REG_GPU = 'true'; % select 'true' for GPU and 'false' for CPU version
params.weights = W; % weights for the PWLS model
params.phantom = Phantom_nbins; % ground truth phantom
params.channel = 3; % selected channel to vis and plot RMSE error
params.ROI = ROI; % phantom region-of-interest
[X_TV, outputTV] = FISTA_REC_rct(params);
figure (2);
subplot(1,3,1); imshow(reshape(X_TV(:,params.channel), n, n), [0 0.1*max(Phantom_nbins(:,params.channel))]); title('FISTA-TV reconstruction of a selected channel');
subplot(1,3,2); plot((outputTV.RMSE(:,params.channel))); title('RMSE error of a selected channel');
subplot(1,3,3); plot((outputTV.obj_func(:,params.channel))); title('Energy functional of a selected channel');
%%
fprintf('%s \n', 'PWLS-dTV-p channel correlated (selection GeoMean) reconstruction using FISTA');
lambda_dTV_geom = 0.0033;
ROI = find((Phantom_nbins(:,1) >= 0));
clear params
params.Amatrix = A; % projection matrix (opTomo operator)
params.sino = sino; % sinogram (vectorized)
params.iterFISTA = 350; %  number of outer iterations (FISTA)
params.REG_method = 'dTV_geom'; % regularization method
params.REG_parameter = lambda_dTV_geom; % dTV - regularization parameter
params.REG_iteration = 50;  % number of inner (proximal) iterations
params.REG_smooth_eta = 0.01; % value to regularise the gradient of the reference image
params.REG_GPU = 'true'; % select 'true' for GPU and 'false' for CPU version
params.weights = W; % weights for the PWLS model
params.probability = snr_geomean; % PDF based on SNR of obtained data
params.phantom = Phantom_nbins; % ground truth phantom
params.channel = 3; % selected channel to vis and plot RMSE error
params.ROI = ROI; % phantom region-of-interest
[X_dTV_geom, outputdTV_geom] = FISTA_REC_rct(params);
figure (3);
subplot(1,3,1); imshow(reshape(X_dTV_geom(:,params.channel), n, n), [0 0.1*max(Phantom_nbins(:,params.channel))]); title('FISTA-dTVp reconstruction of a selected channel');
subplot(1,3,2); plot((outputdTV_geom.RMSE(:,params.channel))); title('RMSE error of a selected channel');
subplot(1,3,3); plot((outputdTV_geom.obj_func(:,params.channel))); title('Energy functional of a selected channel');
%%
fprintf('%s \n', 'PWLS-dTV-d channel correlated (selection Mean k-2:k+2) reconstruction using FISTA');
lambda_dTV_mean = 0.005;
ROI = find((Phantom_nbins(:,1) >= 0));
clear params
params.Amatrix = A; % projection matrix (opTomo operator)
params.sino = sino; % sinogram (vectorized)
params.iterFISTA = 350; %max number of outer iterations
params.REG_method = 'dTV_mean'; % regularization method
params.REG_parameter = lambda_dTV_mean; % dTVd - regularization parameter
params.REG_smooth_eta = 0.05; % value to regularise the gradient of the reference image
params.REG_iteration = 50;  
params.REG_GPU = 'true'; % select 'true' for GPU and 'false' for CPU version
params.weights = W; % weights for the PWLS model
params.phantom = Phantom_nbins; % ground truth phantom
params.channel = 3; % selected channel to vis and plot RMSE error
params.ROI = ROI; % phantom region-of-interest
[X_dTV_mean, outputdTV_mean] = FISTA_REC_rct(params);
figure (4);
subplot(1,3,1); imshow(reshape(X_dTV_mean(:,params.channel), n, n), [0 0.1*max(Phantom_nbins(:,params.channel))]);  title('FISTA-dTVd reconstruction of a selected channel');
subplot(1,3,2); plot((outputdTV_mean.RMSE(:,params.channel))); title('RMSE error of a selected channel');
subplot(1,3,3); plot((outputdTV_mean.obj_func(:,params.channel))); title('Energy functional of a selected channel');
%%
fprintf('%s \n', 'PWLS-TNV reconstruction using FISTA');
lambdaTNV = 0.0005;
ROI = find((Phantom_nbins(:,1) >= 0));
clear params
params.Amatrix = A; % projection matrix (opTomo operator)
params.sino = sino; % sinogram (vectorized)
params.iterFISTA = 350; %max number of outer iterations
params.REG_method = 'TNV'; % regularization method (total nuclear variation)
params.REG_parameter = lambdaTNV; % TNV - regularization parameter
params.REG_iteration = 100; % TNV inner-iterations number
params.weights = W; % weights for the PWLS model
params.phantom = Phantom_nbins; % ground truth phantom
params.channel = 3; % selected channel to vis and plot RMSE error
params.ROI = ROI; % phantom region-of-interest
[X_TNV, outputTNV] = FISTA_REC_rct(params);
figure (5);
subplot(1,2,1); imshow(reshape(X_TNV(:,params.channel), n, n), [0 0.1*max(Phantom_nbins(:,params.channel))]); title('FISTA-TNV reconstruction of a selected channel');
subplot(1,2,2); plot((outputTNV.RMSE(:,params.channel))); title('RMSE error of a selected channel');
%%