close all;clc;clear;

% adding paths to packages
addpath(sprintf(['..' filesep 'SupplementaryPackages' filesep], 1i));
addpath(sprintf(['..' filesep 'SupplementaryPackages' filesep 'spektr' filesep], 1i));
addpath(sprintf(['..' filesep 'SupplementaryPackages' filesep 'spot' filesep], 1i));
addpath(sprintf(['..' filesep 'SupplementaryPackages' filesep 'PhotonAttenuation' filesep], 1i));

pathtodata = '/media/algol/F2FE9B0BFE9AC76F/DATA_KIRILL/BCCloose_1000_3/';
filename = 'input.dis';
filenameData = 'ProjectionData.dis';

% specify dimensions
dimX = 1000;
dimY = 1000;
dimZ = 1000;

% Model parameters
kV =  120;   % voltage
p  =  900;   % number of projections
nd =  round(sqrt(2)*dimX);   % detector pixels

bins = 45:1:115; % the given energy range in KeV
materials = {'SiO2'}; % basis materials

% Fan-beam acquisition geometry (2D)
N0 = 1e5;                % Photon flux (controls noise level)
theta = (0:p-1)*360/p;   % projection angles
dom_width   = 1.0;       % width of domain in cm
src_to_rotc = 3.0;       % dist. from source to rotation center
src_to_det  = 5.0;       % dist. from source to detector
det_width   = 2.0;       % detector width
nbins = length(bins)-1;  % number of energy bins

% Generate source spectrum using Spektr
s = N0*spektrNormalize(spektrSpectrum(kV));

Em = zeros(nbins,1);  % array for mean energy in bins
sb = zeros(nbins,1);  % array for number of photons in each bin
for k = 1:nbins
    I = bins(k):(bins(k+1)-1);
    sk = s(I);
    Em(k) = I*sk/sum(sk);
    sb(k) = sum(sk);
end

mat_density = 2.65; % for SiO2
V = PhotonAttenuation(materials, Em*1e-3, 'mac');  % mass attenuation coef.
Vl = V*diag(mat_density);

%%
% read projection data
fid = fopen(strcat(pathtodata,filenameData),'rb');

Y = fread(fid, p*nd*nbins, 'single');
Y =  single(Y);
Y  = reshape(Y,p,nd,nbins);
Y = -log(Y);
fclose(fid);
%%
vol_geom = astra_create_vol_geom(dimX,dimY);

% Projection geometry (fan-beam)
proj_geom = astra_create_proj_geom('fanflat', dimY*det_width/nd, nd, pi+(pi/180)*theta,...
    dimY*src_to_rotc/dom_width, dimY*(src_to_det-src_to_rotc)/dom_width);

%%
fprintf('%s \n', 'Reconstruction using FBP...');
rec_id = astra_mex_data2d('create', '-vol', vol_geom);
% Set up the parameters for a reconstruction algorithm using the GPU
cfg = astra_struct('FBP_CUDA');
cfg.ReconstructionDataId = rec_id;
cfg.FilterType = 'hamming';
cfg.FilterD = 0.8;

X_FBP = zeros(dimX,dimY,nbins,'single');
% loop through each energy channel (use the previous recon as a prior)
for kk=1:nbins    
       
    % Create projection data object
    proj_id = astra_mex_data2d('create', '-sino', proj_geom, Y(:,:,kk));
    cfg.ProjectionDataId = proj_id;
    
    % Create the algorithm object from the configuration structure
    alg_id = astra_mex_algorithm('create', cfg);
    
    % Run algorithm
    astra_mex_algorithm('iterate', alg_id, 1);
    
    % Get the result
    X_FBP(:,:,kk) = single(astra_mex_data2d('get', rec_id))*2.5e+05;    
  
    % Clean up. Note that GPU memory is tied up in the algorithm object,
    % and main RAM in the data objects.
    astra_mex_algorithm('delete', alg_id);
    astra_mex_data2d('delete', proj_id);   
end  
astra_mex_data2d('delete', rec_id);

%figure; imshow(X_FBP(:,:,5), [0 1]);
%%
% Display reconstructed images
% figure(1); 
% for k = 1:nbins
%     subplot(10,7,k)
%     imagesc(X_FBP(:,:,k),[0,1.0]);
%     title(sprintf('%i-%i',bins(k),bins(k+1)))
% end
%%

