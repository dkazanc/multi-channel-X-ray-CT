% Spectral CT data generation routine using ASTRA toolbox
% depends on ASTRA, SPOT Operator, PhotonAttenuation and Spektr packages

% Run this script just ONCE to generate data 

% This code reproduces numerical results of the paper: 
% Kazantsev D. et al. 2018 
% "Joint image reconstruction method with correlative multi-channel prior for X-ray spectral computed tomography"
% Inverse Problems
% https://doi.org/10.1088/1361-6420/aaba86

close all;clc;clear;

% adding paths to packages
addpath(sprintf(['..' filesep 'SupplementaryPackages' filesep], 1i));
addpath(sprintf(['..' filesep 'SupplementaryPackages' filesep 'spektr' filesep], 1i));
addpath(sprintf(['..' filesep 'SupplementaryPackages' filesep 'spot' filesep], 1i));
addpath(sprintf(['..' filesep 'SupplementaryPackages' filesep 'PhotonAttenuation' filesep], 1i));

% Model parameters
kV =  120;   % voltage
p  =  120;   % number of projections
n  =  512;   % n^2 pixels
nd =  724;   % detector pixels

bins = 45:1:115; % the given energy range in KeV
materials = {'SiO2','Au','Pb','Al'}; % basis materials

% Fan-beam acquisition geometry (2D)
N0 = 4e4;                % Photon flux (controls noise level)
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

% Generate phantom and compute material att. coef. at bin mean energies
[Ut,Vl] = geocore_phantom(n,Em,'phantom');
%% Set up ASTRA volume and projector geometry
fprintf('%s \n', 'Generate measurement data WITHOUT inverse crime...(will take some time)');
factor_d = 2; % factor to upsample the reconstruction grid
[Ut_up,~] = geocore_phantom(n*factor_d,Em,'phantom');

vol_geom_up = astra_create_vol_geom(n*factor_d,n*factor_d);
% Projection geometry (fan-beam)
proj_geom_up_temp = astra_create_proj_geom('fanflat', n*factor_d*det_width/nd, nd, pi+(pi/180)*theta,...
    n*factor_d*src_to_rotc/dom_width, n*factor_d*(src_to_det-src_to_rotc)/dom_width);
proj_id_up_temp = astra_create_projector('strip_fanflat', proj_geom_up_temp, vol_geom_up);

Y_nocrime = zeros(nd*p,nbins);
AUt = zeros(nd*p,size(Ut,3));
for k = 1:size(Ut,3)
    [sinogram_id_temp, sinogram_down_temp] = astra_create_sino(Ut_up(:,:,k), proj_id_up_temp);
    AUt(:,k) = reshape(sinogram_down_temp, nd*p,1)*dom_width/(factor_d*n);
    astra_mex_data2d('delete',sinogram_id_temp);
end
astra_mex_algorithm('delete',proj_id_up_temp);

% Set rng seed
rng(100);
for k = 1:nbins
    Ebin = bins(k):(bins(k+1)-1);
    [~,Vltmp] = geocore_phantom(n, Ebin);
    Y_nocrime(:,k) = Y_nocrime(:,k) + poissrnd(exp(-AUt*Vltmp')*s(Ebin));
end
Y = single(Y_nocrime);

% Display transmission sinograms
figure(1); 
for k = 1:nbins
    subplot(10,7,k)
    imagesc(reshape(Y(:,k)/sb(k),[],nd),[0,1.1]);
    title(sprintf('%i-%i',bins(k),bins(k+1)))
end
% save generated data into mat file
save SpectralData_noCrime.mat Y sb Em p n nd
%%