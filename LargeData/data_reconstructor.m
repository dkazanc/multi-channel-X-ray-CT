close all;clc;clear;

GPU = 'off'; % set to 'on' to use GPU acceleration for ASTRA

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
dimZ = 2000;

% Model parameters
kV =  120;   % voltage
p  =  round(1.2*dimX);   % number of projections
nd =  round(sqrt(2)*dimX);   % detector pixels

bins = 45:35:115; % the given energy range in KeV
materials = {'SiO2'}; % basis materials

% Fan-beam acquisition geometry (2D)
theta = (0:p-1)*360/p;   % projection angles
dom_width   = 1.0;       % width of domain in cm
src_to_rotc = 3.0;       % dist. from source to rotation center
src_to_det  = 3.8;       % dist. from source to detector
det_width   = 2.0;       % detector width

%%
slices = 1;  % How many slices in data? Make it equal to dimZ for whole data

factor_d = 1.0; % factor to downsample [ 0.0 > factor <= 1.0]
dimX_down = round(dimX*factor_d);
dimY_down = round(dimY*factor_d);
vol_geom = astra_create_vol_geom(dimX_down,dimY_down);
% Projection geometry (fan-beam)
proj_geom= astra_create_proj_geom('fanflat', dimY_down*det_width/nd, nd, pi+(pi/180)*theta,dimY_down*src_to_rotc/dom_width, dimY_down*(src_to_det-src_to_rotc)/dom_width);
% Projection geometry (fan-beam) 
 if (strcmp(GPU,'off') == 1)
     proj_id = astra_create_projector('strip_fanflat', proj_geom, vol_geom);
 end
%%
% read projection data
fid = fopen(strcat(pathtodata,filenameData),'rb');
fileSaveRec = strcat('FBPrecon_', 'scale_', num2str(factor_d));
fid_rec = fopen(strcat(pathtodata,fileSaveRec),'wb');

fprintf('%s \n', 'Reconstruction using FBP...');

if (strcmp(GPU,'off') == 1)
    rec_id = astra_mex_data2d('create', '-vol', vol_geom, 0);
    cfg = astra_struct('FBP');    
else 
    % Set up the parameters for a reconstruction algorithm using the GPU
    rec_id = astra_mex_data2d('create', '-vol', vol_geom);
    cfg = astra_struct('FBP_CUDA');
end
    
cfg.ReconstructionDataId = rec_id;
cfg.FilterType = 'hamming';
cfg.FilterD = 0.8;

%
% loop through each energy channel (use the previous recon as a prior)
for kk=1:slices    
    Y = fread(fid, p*nd, 'single');
    Y =  single(Y);
    Y  = reshape(Y,p,nd);
    Y = -log(Y);    
    
    sinogram_id = astra_mex_data2d('create', '-sino', proj_geom, Y);
    cfg.ProjectionDataId = sinogram_id;
    
    % Create projection data object
    if (strcmp(GPU,'off') == 1)        
        cfg.ProjectorId = proj_id;   
    end
    
    % Create the algorithm object from the configuration structure
    alg_id = astra_mex_algorithm('create', cfg);   
    
    % Run algorithm
    astra_mex_algorithm('iterate', alg_id, 1);
    %astra_mex_algorithm('run', alg_id);
    
    % Get the result
    %X_FBP(:,:,kk) = single(astra_mex_data2d('get', rec_id))*(dimX^2);    
    X_FBP = single(astra_mex_data2d('get', rec_id));    
  
    % Clean up. Note that GPU memory is tied up in the algorithm object,
    % and main RAM in the data objects.
    astra_mex_algorithm('delete', alg_id);
    if (strcmp(GPU,'off') == 1)   
    astra_mex_data2d('delete', proj_id); 
    end
    astra_mex_data2d('delete', sinogram_id);     
    
    fwrite(fid_rec, X_FBP, 'single');
end  
astra_mex_data2d('delete', rec_id);
fclose(fid);
fclose(fid_rec);

%%
% read reconstructions:
% fid = fopen(strcat(pathtodata,fileSaveRec),'rb');
% X_FBP = fread(fid, dimX_down*dimY_down*slices, 'single');
% X_FBP =  single(X_FBP);
% X_FBP  = reshape(X_FBP,dimX_down,dimY_down,slices);
% fclose(fid);
% figure; imshow(X_FBP(:,:,2), [0 10]);
%%

