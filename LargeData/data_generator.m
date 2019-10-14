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
dimZ = 2000;

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
% Set up ASTRA volume and projector geometry
% get projection data of a slice
vol_geom = astra_create_vol_geom(dimX,dimY);

% Projection geometry (fan-beam)
proj_geom = astra_create_proj_geom('fanflat', dimY*det_width/nd, nd, pi+(pi/180)*theta,...
    dimY*src_to_rotc/dom_width, dimY*(src_to_det-src_to_rotc)/dom_width);


fid_s = fopen(strcat(pathtodata,filenameData),'wb');
fid = fopen(strcat(pathtodata,filename),'rb');  
%%
slices = 10; % % How many slices in data? Make it equal to dimZ for whole data

% Assuming the fan beam geometry the data will be read slice-by-slice
for i = 1:slices  
    
    slice2D = fread(fid, dimX*dimY, 'uint8');
    slice2D =  single(slice2D);
    slice2D  = reshape(slice2D,dimX,dimY);
    
    [sinogram_id, sino] = astra_create_sino_cuda(slice2D, proj_geom, vol_geom);
    sino = sino*dom_width/(dimX);
    astra_mex_data2d('delete', sinogram_id);
    
    % Set rng seed
    Y = zeros(p,nd,nbins,'single');
    rng(100);
    for k = 1:nbins
        Ebin = bins(k):35:(bins(k+1)-1);
        [~,Vltmp] = geocore_phantom(dimY, Ebin);
        Y(:,:,k) = Y(:,:,k) + poissrnd(exp(-sino*Vltmp(1))*s(Ebin));
        Y(:,:,k) = single(Y(:,:,k)/sb(k));
    end
    
    % summing over all energy bins
    Y_s = sum(Y,3);

    % save projection data
    fwrite(fid_s, Y_s, 'single');
end
fclose(fid);
fclose(fid_s);
%%




