close all;clc;clear;

% adding paths to packages
addpath(sprintf(['..' filesep 'SupplementaryPackages' filesep], 1i));
addpath(sprintf(['..' filesep 'SupplementaryPackages' filesep 'spektr' filesep], 1i));
addpath(sprintf(['..' filesep 'SupplementaryPackages' filesep 'spot' filesep], 1i));
addpath(sprintf(['..' filesep 'SupplementaryPackages' filesep 'PhotonAttenuation' filesep], 1i));

pathtodata = '/media/algol/F2FE9B0BFE9AC76F/DATA_KIRILL/';
filename = 'seg3d5phases.raw';
filenameData = 'ProjectionData.dis';

% specify dimensions
dimX = 200;
dimY = 200;
dimZ = 200;

% Model parameters
kV =  120;   % voltage
p  =  round(1.2*dimX);   % number of projections
nd =  round(sqrt(2)*dimX);   % detector pixels

bins = 35:40:115; % the given energy range in KeV
materials = {'Al2Si2O5OH4', 'SiO2', 'NaAlSi3O8', 'FeS2'}; % basis materials

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

mat_density = [2.6941 2.32 0.9690 7.8600]; % nominal densities
V = PhotonAttenuation(materials, Em*1e-3, 'mac');  % mass attenuation coef.
Vl = V*diag(mat_density);
%%
disp('Loading the whole volume into the memory...');
materials_num = length(materials);
Vol4D = zeros(dimX, dimY, dimZ, materials_num,'single');

for j = 1:materials_num
    fid = fopen(strcat(pathtodata,filename),'rb');  
    for i = 1:dimZ
        slice2D = fread(fid, dimX*dimY, 'uint8');
        slice2D =  single(slice2D);
        slice2D  = reshape(slice2D,dimX,dimY);
        findMat = find(slice2D == j);
        tempSlice = zeros(dimX, dimY, 'single');
        tempSlice(findMat) = 1;
        Vol4D(:,:,i,j) = tempSlice;
    end
    fclose(fid);
end

figure (1);
sliceN = 20; 
subplot(1,4,1); imshow(Vol4D(:,:,sliceN,1), []); title('First phase');
subplot(1,4,2); imshow(Vol4D(:,:,sliceN,2), []); title('Second phase');
subplot(1,4,3); imshow(Vol4D(:,:,sliceN,3), []); title('Third');
subplot(1,4,4); imshow(Vol4D(:,:,sliceN,4), []); title('Fourth');
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
ProjData4D = zeros(det_col_count, length(angles), dimZ, materials_num, 'single');

for k = 1:materials_num
[sino_id, ProjData3D] = astra_create_sino3d_cuda(Vol4D(:,:,:,k), proj_geom, vol_geom);
astra_mex_data3d('delete', sino_id);
ProjData3D = single(ProjData3D)/(dimX*5);
ProjData4D(:,:,:,k) = ProjData3D;
clear ProjData3D;    
end

figure (2);
sliceN = 20; 
subplot(1,4,1); imshow(ProjData4D(:,:,sliceN,1), []); title('First phase');
subplot(1,4,2); imshow(ProjData4D(:,:,sliceN,2), []); title('Second phase');
subplot(1,4,3); imshow(ProjData4D(:,:,sliceN,3), []); title('Third');
subplot(1,4,4); imshow(ProjData4D(:,:,sliceN,4), []); title('Fourth');
%%
disp('Adding Poisson noise and perform data normalisation, will take time!');
Y = zeros(det_col_count, length(angles), dimZ, nbins, 'single');
rng(100);

for k = 1:nbins
    Ebin = bins(k):40:(bins(k+1)-1);
    [~,Vltmp] = geocore_phantom(dimY, Ebin);
    for j = 1:dimZ
    proj_resh = reshape(ProjData4D(:,:,j,:),det_col_count*length(angles), 4);
    pois_res = poissrnd(exp(-proj_resh*Vltmp')*s(Ebin));
    pois_res = reshape(pois_res,det_col_count,length(angles));    
    Y(:,:,j,k) = pois_res;
    Y(:,:,j,k) = single(Y(:,:,j,k)/sb(k));
    end
end
% summing over all energy bins
Y_s = sum(Y,4);
Y = -log(Y_s);
max(Y(:))
figure; imshow(squeeze(Y(:,:,100)), []);
clear Y_s ProjData4D pois_res;
% figure; imshow(squeeze(Y(:,:,5)), []);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
disp('Reconstruction of cone beam data...');
factor_min = 0.05; % the initial downsampling size
factor_max = 1.0; % the max size (1.0 equals to the original size)
downsample_num = 1; % the totalnumber of downsampling levels
all_down_factors = linspace(factor_min, factor_max, downsample_num);

for n = 1:downsample_num
    factor = all_down_factors(n);    
    dimX_newsize = round(dimX*factor);
    dimY_newsize = round(dimY*factor);
    dimZ_newsize = round(dimZ*factor);

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
    
    % Save the reconstructed 3D data
    filenameRecon = strcat('FDK_recon_',num2str(dimX_newsize),'_',num2str(dimY_newsize),'_', num2str(dimZ_newsize));
    fid_s = fopen(strcat(pathtodata,filenameRecon),'wb');
    for i = 1:dimZ_newsize      
        % save projection data
        fwrite(fid_s, reconstr3D(:,:,i), 'single');
    end
    fclose(fid_s);
end

figure; imshow(reconstr3D(:,:,10), [ ]);
%%
% % Save generated 3D cone beam data into a file to reuse later on
% fid_s = fopen(strcat(pathtodata,filenameData),'wb');
% for i = 1:dimZ      
%     % save projection data
%     fwrite(fid_s, Y(:,:,i), 'single');
% end
% fclose(fid_s);
%%
% reading the reconstructed (saved) data
% pathtodata = '/media/algol/F2FE9B0BFE9AC76F/DATA_KIRILL/BCCclose_2000/';
% filename = 'FDK_recon_1050_1050_5';
% dimX = 1050;
% dimY = 1050;
% dimZ = 5;
% 
% fid = fopen(strcat(pathtodata,filename),'rb');  
% vol3D = zeros(dimX,dimY,dimZ,'single');
% 
% for i = 1:dimZ  
%     
%     slice2D = fread(fid, dimX*dimY, 'single');
%     slice2D =  single(slice2D);
%     slice2D  = reshape(slice2D,dimX,dimY);
%     vol3D(:,:,i) = slice2D;
% end
% fclose(fid);
%%

