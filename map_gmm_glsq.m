function [X_hat,Y_hat] = map_gmm_glsq(Z,gmm,given_comps,X_hat,Y_hat)

im_sz = size(Z);
patch_p = round(sqrt(gmm.dim));
patch_sz = [patch_p,patch_p];
if isempty(given_comps)
  given_comps.X = nan(size(Z));
  given_comps.Y = nan(size(Z));
end

W = speye(numel(Z))*1e-6; % TODO: figure out why this speeds up backslash
Q = speye(numel(Z))*1e-6; %       and matrix multiplication

given_comps_X = ~isnan(given_comps.X);
given_comps_Y = ~isnan(given_comps.Y);

global glsq_precomp;
if isempty(glsq_precomp) || ~isequal(glsq_precomp.im_sz,im_sz)
  [glsq_precomp.T_nnz,glsq_precomp.patch_inds] = prepare_T_nnz(im_sz,patch_sz);
  glsq_precomp.T_inds = prepare_T_inds(glsq_precomp.T_nnz,glsq_precomp.patch_inds,im_sz,patch_sz);
  glsq_precomp.im_sz = im_sz;
end

X_hat_col = im2col(reshape(X_hat,im_sz),patch_sz,'sliding'); % TODO: should this be sped-up?
Y_hat_col = im2col(reshape(Y_hat,im_sz),patch_sz,'sliding');

% e-step
[~,comps_X] = max(calc_resp(X_hat_col,gmm)); % TODO: currently only hard-filtering supported
[~,comps_Y] = max(calc_resp(Y_hat_col,gmm));
comps_X(given_comps_X) = given_comps.X(given_comps_X);
comps_Y(given_comps_Y) = given_comps.Y(given_comps_Y);

% m-step
[T1,T2] = prepare_T1_T2(glsq_precomp.T_nnz,glsq_precomp.T_inds,im_sz,gmm.invcovs,...
  comps_X,comps_Y,given_comps_X,given_comps_Y);
lhs = (T1 + T2) + W;
rhs = (T2 + Q)*Z(:);
X_hat = lhs\rhs;
Y_hat = Z(:)-X_hat;    


function [T_nnz,patch_inds] = prepare_T_nnz(im_sz,patch_sz)

num_patch_pixels = prod(patch_sz);
num_im_pixels = prod(im_sz);

T_nnz = spalloc(num_im_pixels,num_im_pixels,num_im_pixels*prod(patch_sz)^2);
patch_inds = zeros(num_patch_pixels,im_sz(1)-patch_sz(1)+1,im_sz(2)-patch_sz(2)+1);

for j=1:size(patch_inds,3)
  for i=1:size(patch_inds,2)
    [ind_j,ind_i] = meshgrid(j:j+patch_sz(2)-1, i:i+patch_sz(1)-1);
    inds = sub2ind(im_sz,ind_i,ind_j);
    patch_inds(:,i,j) = inds(:);
    T_nnz(inds(:),inds(:)) = 1;
  end
end


function T_inds = prepare_T_inds(T_nnz,patch_inds,im_sz,patch_sz)

num_patch_pixels = prod(patch_sz);
num_im_pixels = prod(im_sz);

[ii,jj,tt] = find(T_nnz);
T_nnz_ = sparse(ii,jj,(1:numel(tt))',num_im_pixels,num_im_pixels);
T_inds = zeros(num_patch_pixels,num_patch_pixels,im_sz(1)-patch_sz(1)+1,im_sz(2)-patch_sz(2)+1);

for j=1:size(patch_inds,3)
  for i=1:size(patch_inds,2)
    inds = patch_inds(:,i,j);
    T_inds(:,:,i,j) = T_nnz_(inds,inds);
  end
end
T_inds = reshape(T_inds,num_patch_pixels,num_patch_pixels,size(T_inds,3)*size(T_inds,4));


function [T1,T2] = prepare_T1_T2(T_nnz,T_inds,im_sz,invcovs,comps_X,comps_Y,given_comps_X,given_comps_Y)

zeta = [1,1e3];
num_im_pixels = prod(im_sz);
[ii,jj,tt] = find(T_nnz);
tt1 = tt*1e-10;
tt2 = tt*1e-10;
for k=1:size(T_inds,3)
  inds = T_inds(:,:,k);
  tt1(inds) = tt1(inds) + invcovs(:,:,comps_X(k))*zeta(1+given_comps_X(k));
  tt2(inds) = tt2(inds) + invcovs(:,:,comps_Y(k))*zeta(1+given_comps_Y(k));
end
T1 = sparse(ii,jj,tt1,num_im_pixels,num_im_pixels);
T2 = sparse(ii,jj,tt2,num_im_pixels,num_im_pixels);


function logPXZ = calc_resp(X,gmm)

if (isfield(gmm,'net')) % posterior according to gating network if available
  logPXZ = permute(gmm.net.forward(gmm.net,permute(X,[1,3,4,5,2])),[1,5,2,3,4]);

else % direct evaluation of posterior
  logPXZ = zeros(gmm.nmodels,size(X,2));
  parfor i=1:gmm.nmodels
    logPXZ(i,:) = log(gmm.mixweights(i)) + log_gauss_pdf(X,gmm.covs(:,:,i));
  end
end