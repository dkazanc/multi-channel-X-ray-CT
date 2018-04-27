function [U,Vl,varargout] = geocore_phantom(N,E,switcher)

if exist('switcher','var')
    
else 
    switcher = 'materials';
end

materials   = {'Quartz','Pyrite','Galena','Gold'};
mat_short   = {'SiO2','Fe(0.466)S(0.534)','Pb(0.866)S(0.134)','Au'};
mat_density = [2.65 5.01 7.6 19.25];

V = PhotonAttenuation(mat_short, E*1e-3, 'mac');  % mass attenuation coef.
Vl = V*diag(mat_density);

if (strcmp(switcher, 'phantom') == 1)
% load 2D spectral phantoms which consists of 4 materials.
% The phantom has been built using TomoPhantom software:
% http://doi.org/10.5281/zenodo.1215759    
  if (N == 512) 
      load SpectralPhantom512.mat;
  elseif (N == 1024) 
       load SpectralPhantom1024.mat;
  else
      fprintf('%s \n', 'Only 512 and 1024 dimensions are acceptable');
  end
  
else 
U = 0;
end

if nargout > 2
    varargout(1) = {materials};
end

end