function [X, outputF] = FISTA_REC_rct(params)

% FISTA multi-channel reconstruction approach using pre-built SPOT operator A (proj. matrix)
% This code reproduces results of the paper: 
% Kazantsev D. et al. 2018 
% "Joint image reconstruction method with correlative multi-channel prior for X-ray spectral computed tomography"
% Inverse Problems
% https://doi.org/10.1088/1361-6420/aaba86

% ___Input___:
% params.[] file:
%       - .A (projection matrix formed by SPOT operator) [required]
%       - .sino (vectorised 3D sinogram where 3-dim is energy) [required]
%       - .iterFISTA (outer FISTA iterations)
%       - .L_const (Lipschitz constant, default - Power method)                                                                                                    )
%       - .phantom (ground truth image)
%       - .ROI (Region-of-interest, only if phantom is provided)
%       - .weights (statisitcal weighting, sinogram size, default 1)
%       - .tol (tolerance to terminate TV regularization, default 1.0e-04)
%       - .REG_method (choose between 'TV','dTV_geom','dTV_mean' or 'TNV')
%       - .REG_parameter (regularization parameter, default 0 - regul is switched off)
%       - .REG_iteration (total number of iterations)
%       - .REG_GPU (choose between 'true' or 'false' (CPU power by default))
%       - .channel (to visualize a selected channel)
% ___Output___:
% 1. X - reconstructed energy-channels
% 2. outputF:
%           obj_func - objective function value
%           error_vec - vector of RMSE errors for each channel
%           L_const - Lipschitz constant per channel
%           X_select - selected channels with respect to lower RMSE

if (isfield(params,'Amatrix'))
    A = params.Amatrix;
else
    error('%s \n', 'Please provide the projection matrix ');
end
if (isfield(params,'sino'))
    B = params.sino;
else
    error('%s \n', 'Please provide a sinogram');
end
n_vox = size(A,2);
N = sqrt(n_vox);
num_channels = size(B,2);
if (num_channels > 1)
    fprintf('%s %i %s \n', 'Given sinogram has', num_channels, 'slices (energy-channels)');
end
if (isfield(params,'weights'))
    W = params.weights;
else
    W = ones(size(B));
end
if (isfield(params,'L_const'))
    L_const = params.L_const;
else
    % using Power method (PM) to establish L constant
    niter = 15; % number of iteration for PM
    xtmp = rand(n_vox,num_channels);
    for k = 1:niter
        xtmp = bsxfun(@rdivide,A'*(W.*(A*xtmp)),sqrt(sum(xtmp.^2)));
    end
    L_const = min(sqrt(sum(xtmp.^2)));
    clear xtmp;
end
if (isfield(params,'iterFISTA'))
    iterFISTA = params.iterFISTA;
else
    iterFISTA = 30;
end
if (isfield(params,'REG_parameter'))
    lambda = params.REG_parameter;
else
    lambda = 0.001;
end
if (isfield(params,'REG_tol'))
    tol = params.REG_tol;
else
    tol = 1.0e-05;
end
if (isfield(params,'REG_smooth_eta'))
    eta_val = params.REG_smooth_eta;
else
    eta_val = 0.01;
end
if (isfield(params,'REG_iteration'))
    Iter_InnerProx = params.REG_iteration;
else
    Iter_InnerProx = 150;
end
if (isfield(params,'REG_print'))
    REG_print = params.REG_print;
else
    REG_print = 0;
end
if (isfield(params,'REG_method'))
    if (strcmp('TV',params.REG_method) == 1)
        fprintf('%s \n', 'FISTA-TV reconstruction method is selected');
    elseif (strcmp('dTV_geom',params.REG_method) == 1)
        fprintf('%s \n', 'FISTA-dTV method with GeoMean channel selection');
    elseif (strcmp('dTV_mean',params.REG_method) == 1)
        fprintf('%s \n', 'FISTA-dTV method with k-2:k+2 channel averaging is selected');
    elseif (strcmp('TNV',params.REG_method) == 1)
        fprintf('%s \n', 'FISTA-TNV reconstruction method is selected');
    else
        error('%s \n', 'Please select TV, dTV_geom, dTV_mean or TNV method');
    end
else
    fprintf('%s \n', 'FISTA reconstruction method selected');
end
if (isfield(params,'phantom'))
    phantom = params.phantom;
    switchphant = 1;
else
    switchphant = 0;
end
if (isfield(params,'channel'))
    channel = params.channel;
else
    channel = 1;
end
if (isfield(params,'initializ'))
    X = params.initializ;
else
    X = zeros(n_vox,num_channels); % storage for the solution
end
if (isfield(params,'ROI'))
    ROI = params.ROI;
else
    ROI = find(X(:,1)>=0);
end
if (isfield(params,'nonneg'))
    nonneg = params.nonneg;
else
    nonneg = 0;
end
if (isfield(params,'probability'))
    probability = params.probability;
else
    probability = ones(num_channels,1);
end
if (isfield(params,'Optim_channels'))
    OptimalChannels = params.Optim_channels;
    X_select = zeros(n_vox,num_channels);
else
    OptimalChannels = 0;
    X_select = 0;
end
if (isfield(params,'maxvalplot'))
    maxvalplot = params.maxvalplot;
else
    maxvalplot = 6;
end
if (isfield(params,'REG_GPU'))
    REG_GPU = params.REG_GPU;
else
    REG_GPU = 'false';
end

% FISTA-based recovery using projection matrix A
error_vec = zeros(iterFISTA, num_channels);
obj_func_value = zeros(iterFISTA, num_channels);

Y = X;
if ~isnan(channel)
    figure(10);
end
t = ones(1,num_channels);

% FISTA main outer loop
for i = 1:iterFISTA
    X_old = X;
    t_old = t;
    
    resT = W.*(A*Y - B);
    X = Y - bsxfun(@rdivide, A'*(resT), L_const);
    
    % calculate data-term objective value
    obj_func_value_temp = (0.5*sum(resT(:).^2)); % for the objective function output
    
    if (lambda > 0)
        if (strcmp('TV',params.REG_method) == 1)
            for jj = 1:num_channels
                X_resh = reshape(single(X(:,jj)), N,N);
                % FGP-TV minimization subproblem
                lambda_reg = lambda/L_const(jj);
                if (strcmp('true', REG_GPU) == 1)
                    % GPU version
                    X_den = FGP_TV_GPU(single(X_resh), lambda_reg, Iter_InnerProx, tol, 'iso', nonneg, REG_print);                    
                else
                    % CPU version
                    X_den = FGP_TV(X_resh, lambda_reg, Iter_InnerProx, tol, 'iso', nonneg, REG_print);             
                end
            f_val = TV_energy(X_den, X_resh, lambda(jj), 2);  % get energy function of the TV penalty
            % Store the updated objective value 
            obj_func_value(i,jj) = obj_func_value_temp + f_val;
            X(:,jj) = reshape(X_den, n_vox, 1);
            end                                  
            
        elseif (strcmp('dTV_geom',params.REG_method) == 1)
            % FGP-dTVp minimization subproblem with the geometric mean based selection of channels
            draw = gendist(probability,1,num_channels); % draw a channel (integer) according to the given PDF
            for jj = 1:num_channels
                chC = draw(jj); %randomly drawn channel
                supp_channel = reshape(single(X_old(:,chC)), N,N);
                lambda_reg = lambda/L_const;
                X_resh = reshape(single(X(:,jj)), N,N);                
                if (strcmp('true', REG_GPU) == 1)
                    % GPU version
                    X_den = FGP_dTV_GPU(single(X_resh), single(supp_channel), lambda_reg, Iter_InnerProx, tol, eta_val, 'iso', nonneg, REG_print);  
                else
                    % CPU version
                    X_den = FGP_dTV(single(X_resh), single(supp_channel), lambda_reg, Iter_InnerProx, tol, eta_val, 'iso', nonneg, REG_print);  
                end
            f_val = TV_energy(X_den, X_resh, lambda, 2);  % get energy function of the TV penalty
            % Store the updated objective value 
            obj_func_value(i,jj) = obj_func_value_temp + f_val;
            X(:,jj) = reshape(X_den, n_vox, 1);
            end
        elseif (strcmp('dTV_mean',params.REG_method) == 1)
            % FGP-dTV minimization subproblem with k-2:k+2 average selection of channels            
            meanCh = zeros(N,N,5,'single');
            for jj = 1:num_channels
                counterCh = -2;
                for z1 = 1:5
                    if (((jj+counterCh) < 1) || ((jj+counterCh) > num_channels))
                        chn_add2meanCh = zeros(N,N,'single');
                    else
                        chn_add2meanCh = reshape(single(X_old(:,(jj+counterCh))), N,N);
                    end
                    meanCh(:,:,z1) = chn_add2meanCh;
                    counterCh = counterCh + 1;
                end
                supp_channel = mean(meanCh,3);
                lambda_reg = lambda/L_const;
                X_resh = reshape(single(X(:,jj)), N,N); 
                if (strcmp('true', REG_GPU) == 1)
                    % GPU version
                    X_den = FGP_dTV_GPU(single(X_resh), single(supp_channel), lambda_reg, Iter_InnerProx, tol, eta_val, 'iso', nonneg, REG_print);
                else
                    % CPU version                    
                    X_den = FGP_dTV(single(X_resh), single(supp_channel), lambda_reg, Iter_InnerProx, tol, eta_val, 'iso', nonneg, REG_print);  
                end
            f_val = TV_energy(X_den, X_resh, lambda, 2);  % get energy function of the TV penalty
            % Store the updated objective value 
            obj_func_value(i,jj) = obj_func_value_temp + f_val;
            X(:,jj) = reshape(X_den, n_vox, 1);
            end
        elseif (strcmp('TNV',params.REG_method) == 1)
            % regularization using Total Nuclear variation from the
            % Coloborative Total Variation package (see work of J. Duran)
            X_den = zeros(N,N,num_channels,'single');
            for jj = 1:num_channels
                X_den(:,:,jj) = reshape(single(X(:,jj)), N,N);
            end
            lambda_reg = lambda/L_const;
            tic; X_den = TNV(single(X_den), lambda_reg, Iter_InnerProx, tol); toc;
            for jj = 1:num_channels
                X(:,jj) = reshape(X_den(:,:,jj), n_vox,1);
                % obj_func_value(i,jj) = obj_func_value(i,jj) + f_val(jj);
            end
        end
    end
    t = (1 + sqrt(1 + 4.*t.^2))./2;    
    Y = X +  bsxfun(@rdivide,(X - X_old), (t./(t_old - 1)));
    
    if (switchphant == 1)
        for jj = 1:num_channels
            ResErrors = CalcPerf(phantom(ROI,jj),X(ROI,jj));
            error_vec(i, jj) = ResErrors.RMSE.*100;
        end
    end
    
    if (OptimalChannels(1) > 0)
        % saving channels with the lowest RMSE
        ChannelSel = find(OptimalChannels == i, num_channels);
        numCh = length(ChannelSel);
        if (numCh > 0)
            for jj = 1:numCh
                X_select(:,ChannelSel(jj)) = X(:,ChannelSel(jj));
            end
        end
    end
    
    if ~isnan(channel)
        imshow(reshape(X(:,channel), N,N), [0 maxvalplot]); pause(0.05);
    end
    if (switchphant == 1)
        fprintf('%s %i %s %s %i %s %f \n', '>>>>> FISTA iteration', i, ';', 'RMSE error of channel', channel, 'equals to', error_vec(i, channel));
    else
        fprintf('%s %i \n', 'FISTA iteration', i);
    end
end

outputF.RMSE = error_vec;
outputF.obj_func = obj_func_value;
outputF.L_const = L_const;
outputF.X_optimal = X_select;