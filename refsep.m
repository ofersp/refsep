function [X_hat,Y_hat,res_psnr,given_comps] = refsep(Z,X,Y,params)

patch_sz = [8,8];

if ~exist('X','var'); X = []; end
if ~exist('Y','var'); Y = []; end
if ~exist('params','var'); params = struct(); end
have_gt = ~isempty(X) && ~isempty(Y);

% params and random seed
if ~isfield(params,'load_annot_file'); params.load_annot_file = []; end
if ~isfield(params,'use_gating'); params.use_gating = false; end
if ~isfield(params,'num_glsq_iters'); params.num_glsq_iters = 6; end
if ~isfield(params,'show_progress'); params.show_progress = false; end
if ~isfield(params,'show_result'); params.show_result = (nargout() == 0); end

% load pre-trained model and gating net
global gmm;
if isempty(gmm)
  load('models/gmm.mat');
end
if params.use_gating
  load('models/net.mat');
  gmm.net = net; clear net;
else
  if isfield(gmm,'net'); gmm = rmfield(gmm,'net'); end
end

% get user annotations
if isempty(params.load_annot_file)
  given_comps = get_annot_comps_user(Z,gmm);
else
  tmp = load(params.load_annot_file);
  given_comps = given_comps_im2col(tmp.given_comps_im, patch_sz); 
  clear tmp;
end

% decompose a single channel
[X_hat,Y_hat,res_mse] = decompose_channel(Z,X,Y,params,gmm,given_comps);
if have_gt
  res_psnr = -10*log10(mean(res_mse)); 
else
  res_psnr = [];
end
   
% show result
if params.show_result
  figure(1);
  subplot(1,2,1); imagesc(X_hat+0.5); colormap gray; axis image; axis off;
  subplot(1,2,2); imagesc(Y_hat+0.5); colormap gray; axis image; axis off; 
  drawnow;
  if ~isempty(res_psnr)
    fprintf(1,'resulting PSNR: %0.3f\n', res_psnr);
  end
end

function [X_hat,Y_hat,res_mse] = decompose_channel(Z,X,Y,params,gmm,given_comps)

% decompose
[X_hat,Y_hat] = refsep_glsq(Z/2,Z,params.num_glsq_iters,gmm,given_comps,params.show_progress);

% recover means and measure accuracy when ground truth is available
res_mse = [];
if ~isempty(X) && ~isempty(Y)
  X_hat = X_hat - mean(X_hat(:)) + mean(X(:));
  Y_hat = Y_hat - mean(Y_hat(:)) + mean(Y(:));
  res_mse = mse(X_hat, X);
end