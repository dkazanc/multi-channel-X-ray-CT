close all;clc;clear;

% adding paths to packages
addpath(sprintf(['..' filesep 'SupplementaryPackages' filesep], 1i));
addpath(sprintf(['..' filesep 'SupplementaryPackages' filesep 'spektr' filesep], 1i));
addpath(sprintf(['..' filesep 'SupplementaryPackages' filesep 'spot' filesep], 1i));
addpath(sprintf(['..' filesep 'SupplementaryPackages' filesep 'PhotonAttenuation' filesep], 1i));

%specify materials
materials = {'Air','Al2Si2O5OH4', 'SiO2', 'NaAlSi3O8', 'FeS2'}; % basis materials
materials_num = length(materials);
mat_density = [0.1 1.3 2.32 2.65 7.8600]; % nominal densities 

% specify dimensions
dimX = 1000;
dimY = 1000;
dimZ = 3;
dimX_pad = 1200;
dimY_pad = 1200;
dimZ_pad = 3;
padZ = (dimZ_pad - dimZ)/2;


pathtodata = '/home/algol/Documents/MATLAB/multi-channel-X-ray-CT/LargeData/';
filename = 'original_1000.raw';
filenameData = 'ProjectionData.dis';

% Vol4D = zeros(dimX_pad, dimY_pad, dimZ_pad, materials_num,'single');
% disp('Loading the whole volume into the memory...');
% for j = 1:materials_num
%     fid = fopen(strcat(pathtodata,filename),'rb');  
%     for i = padZ+1:(dimZ_pad-padZ)
%         slice2D = fread(fid, dimX*dimY, 'uint8');
%         slice2D =  single(slice2D);
%         slice2D  = reshape(slice2D,dimX,dimY);
%         findMat = find(slice2D == j-1);
%         tempSlice = zeros(dimX, dimY, 'single');
%         tempSlice(findMat) = 1;
%         if (j == 0) 
%             Vol4D(:,:,i,j) = padarray(tempSlice,[100 100],1);
%         else
%             Vol4D(:,:,i,j) = padarray(tempSlice,[100 100],0);
%         end
%     end
%     fclose(fid);
% end

load('vol4d_5materials.mat'); % load the data

dimX = dimX_pad;
dimY = dimY_pad;
dimZ = dimZ_pad;

% Model parameters
kV =  120;   % voltage
p  =  round(1.6*dimX);   % number of projections
nd =  round(sqrt(2)*dimX);   % detector pixels

% Cone-beam acquisition geometry (3D)
N0 = 5e5;                % Photon flux (controls noise level)
theta = (0:p-1)*360/p;   % projection angles
dom_width   = 1.0;       % width of domain in cm
src_to_rotc = 4.0;       % dist. from source to rotation center
src_to_det  = 3.8;       % dist. from source to detector
det_width   = 2.0;       % detector width

bins = 20:2:115; % the given energy range in KeV
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

%%
figure(1);
sliceN = 2; 
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
ProjData3D = single(ProjData3D)/(dimX);
ProjData4D(:,:,:,k) = ProjData3D;
clear ProjData3D;    
end

figure (2);
sliceN = 2; 
subplot(1,4,1); imshow(ProjData4D(:,:,sliceN,1), []); title('First phase');
subplot(1,4,2); imshow(ProjData4D(:,:,sliceN,2), []); title('Second phase');
subplot(1,4,3); imshow(ProjData4D(:,:,sliceN,3), []); title('Third');
subplot(1,4,4); imshow(ProjData4D(:,:,sliceN,4), []); title('Fourth');
%%
disp('Adding Poisson noise and perform data normalisation, will take time!');
Y = zeros(det_col_count, length(angles), dimZ, nbins, 'single');
rng(100);

for k = 1:nbins
    V = PhotonAttenuation(materials, Em(k)*1e-3, 'mac');  % mass attenuation coef.
    Vltmp = V*diag(mat_density);    
    for j = 1:dimZ
    proj_resh = reshape(ProjData4D(:,:,j,:),det_col_count*length(angles), materials_num);
    calibr_poisson = poissrnd(sb(k)*ones(size(proj_resh,1),1)); % adding noise to a constant calibration value 
    pois_res = poissrnd(exp(-proj_resh*Vltmp').*calibr_poisson); % adding noise to calibrated data 
    pois_res = reshape(pois_res,det_col_count,length(angles));        
    Y(:,:,j,k) = -log(bsxfun(@times, pois_res+(pois_res==0), 1./sb(k)));
    end
end
Y = mean(Y,4); % summing over all energy bins
figure(3); imshow(squeeze(Y(:,:,2)), []);
clear Y_s ProjData4D pois_res;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
disp('Reconstruction of cone beam data...');
all_down_factors = 1.0; %[0.1 , 0.125, 0.2  , 0.25 , 0.5  , 1.];
downsample_num = length(all_down_factors);

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
%     filenameRecon = strcat('FDK_recon_',num2str(dimX_newsize),'_',num2str(dimY_newsize),'_', num2str(dimZ_newsize));
%     fid_s = fopen(strcat(pathtodata,filenameRecon),'wb');
%     for i = 1:dimZ_newsize      
%         % save projection data
%         fwrite(fid_s, reconstr3D(:,:,i), 'single');
%     end
%     fclose(fid_s);
end

figure; imshow(reconstr3D(:,:,2), [0.0 1.5]);
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