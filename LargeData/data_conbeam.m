close all;clc;clear;

% adding paths to packages
addpath(sprintf(['..' filesep 'SupplementaryPackages' filesep], 1i));
addpath(sprintf(['..' filesep 'SupplementaryPackages' filesep 'spektr' filesep], 1i));
addpath(sprintf(['..' filesep 'SupplementaryPackages' filesep 'spot' filesep], 1i));
addpath(sprintf(['..' filesep 'SupplementaryPackages' filesep 'PhotonAttenuation' filesep], 1i));

pathtodata = '/media/algol/F2FE9B0BFE9AC76F/DATA_KIRILL/BCCclose_2000/';
filename = 'input.dis';
filenameData = 'ProjectionData.dis';

% specify dimensions
dimX = 2000;
dimY = 2000;
dimZ = 15;

% Model parameters
kV =  120;   % voltage
p  =  round(1.2*dimX);   % number of projections
nd =  round(sqrt(2)*dimX);   % detector pixels

bins = 45:35:115; % the given energy range in KeV
materials = {'SiO2'}; % basis materials

% Fan-beam acquisition geometry (2D)
N0 = 5e5;                % Photon flux (controls noise level)
theta = (0:p-1)*360/p;   % projection angles
dom_width   = 1.0;       % width of domain in cm
src_to_rotc = 3.0;       % dist. from source to rotation center
src_to_det  = 3.8;       % dist. from source to detector
det_width   = 2.0;       % detector width
nbins = length(bins)-1;  % number of energy bins

% Generate source spectrum using Spektr
s = N0*spektrNormalize(spektrSpectrum(kV));

Em = zeros(nbins,1);  % array for mean energy in bins
sb = zeros(nbins,1);  % array for number of photons in each bin
for k = 1:nbins
    I = bins(k):(bins(k+1));
    sk = s(I);
    Em(k) = I*sk/sum(sk);
    sb(k) = sum(sk);
end

mat_density = 2.65; % for SiO2
V = PhotonAttenuation(materials, Em*1e-3, 'mac');  % mass attenuation coef.
Vl = V*diag(mat_density);

%%
disp('Loading the whole volume into the memory...');
fid = fopen(strcat(pathtodata,filename),'rb');  
vol3D = zeros(dimX,dimY,dimZ,'single');

for i = 1:dimZ  
    
    slice2D = fread(fid, dimX*dimY, 'uint8');
    slice2D =  single(slice2D);
    slice2D  = reshape(slice2D,dimX,dimY);
    vol3D(:,:,i) = slice2D;
end
fclose(fid);
%%
% Set up ASTRA volume and projector cone beam geometry (GPU only)
vol_geom = astra_create_vol_geom(dimX, dimY, dimZ);

det_spacing_x = dimY*det_width/nd;
det_spacing_y = dimX*det_width/nd;
det_row_count = dimZ;
det_col_count = nd; 
angles = theta*(pi/180);
source_origin = dimY*src_to_rotc/dom_width;
origin_det = dimY*(src_to_det-src_to_rotc)/dom_width;

proj_geom = astra_create_proj_geom('cone',  det_spacing_x, det_spacing_y, det_row_count, det_col_count, angles, source_origin, origin_det);

disp('Generating projection data...');
[sino_id, ProjData3D] = astra_create_sino3d_cuda(vol3D, proj_geom, vol_geom);
astra_mex_data3d('delete', sino_id);
ProjData3D = single(ProjData3D)/(dimX);
figure; imshow(squeeze(ProjData3D(:,:,5)), []);
%%
disp('Remove the original volume to free the memory!');
clear vol3D
%%
disp('Adding Poisson noise and perform data normalisation (very memory hungry), will take time!');
Y = zeros(det_col_count,length(angles),dimZ,nbins,'single');
rng(100);
for k = 1:nbins
    Ebin = bins(k):35:(bins(k+1)-1);
    [~,Vltmp] = geocore_phantom(dimY, Ebin);
    Y(:,:,:,k) = Y(:,:,:,k) + poissrnd(exp(-(ProjData3D)*Vltmp(1))*s(Ebin));
    Y(:,:,:,k) = single(Y(:,:,:,k)/sb(k));
end
% summing over all energy bins
Y_s = sum(Y,4);
Y = -log(Y_s); clear Y_s ProjData3D slice2D;
figure; imshow(squeeze(Y(:,:,5)), []);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
disp('Reconstruction of cone beam data...');
factor_down = 1.0; % change it to a smaller value for downsampled reconstruction
dimX_newsize = round(dimX*factor_down);
dimY_newsize = round(dimY*factor_down);
dimZ_newsize = round(dimZ*factor_down);

vol_geom = astra_create_vol_geom(dimX_newsize, dimY_newsize, dimZ_newsize);

det_spacing_x_new = dimY_newsize*det_width/nd;
det_spacing_y_new = dimX_newsize*det_width/nd;
source_origin_new = dimY_newsize*src_to_rotc/dom_width;
origin_det_new = dimY_newsize*(src_to_det-src_to_rotc)/dom_width;

% Projection geometry (cone-beam)
proj_geom = astra_create_proj_geom('cone',  det_spacing_x_new, det_spacing_y_new, det_row_count, det_col_count, angles, source_origin_new, origin_det_new);

cfg = astra_struct('FDK_CUDA');
reconstruction_id = astra_mex_data3d('create', '-vol', vol_geom, 0.0);
sinogram_id = astra_mex_data3d('create', '-sino', proj_geom, Y);
cfg.ProjectionDataId = sinogram_id;
cfg.ReconstructionDataId = reconstruction_id;

alg_id = astra_mex_algorithm('create', cfg);       
% Run algorithm
astra_mex_algorithm('iterate', alg_id, 1);
reconstr3D = single(astra_mex_data3d('get', reconstruction_id));  
reconstr3D = reconstr3D*(0.5*dimX);

astra_mex_data3d('delete', sinogram_id);
astra_mex_data3d('delete', alg_id);
astra_mex_data3d('delete', reconstruction_id);

figure; imshow(reconstr3D(:,:,3), [0 2.0]);
%%
% Save generated 3D cone beam data into a file to reuse later on
fid_s = fopen(strcat(pathtodata,filenameData),'wb');
for i = 1:dimZ      
    % save projection data
    fwrite(fid_s, Y(:,:,i), 'single');
end
fclose(fid_s);
%%
% Save the reconstructed 3D data
filenameRecon = strcat('FDK_recon_',num2str(dimX_newsize),'_',num2str(dimY_newsize),'_', num2str(dimZ_newsize));
fid_s = fopen(strcat(pathtodata,filenameRecon),'wb');
for i = 1:dimZ_newsize      
    % save projection data
    fwrite(fid_s, reconstr3D(:,:,i), 'single');
end
fclose(fid_s);
%%

