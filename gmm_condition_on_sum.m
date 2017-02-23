function gmm_c = gmm_condition_on_sum(gmm,Z)

d = gmm.dim;
n = gmm.nmodels;

gmm_c.dim = d;
gmm_c.nmodels = n^2;

global gmm_c_pre;
if isempty(gmm_c_pre)
  gmm_c_pre.covs = zeros(d,d,n,n);
  gmm_c_pre.denom = zeros(n,n);
  H_t = zeros(d,d,n,n);
  inv_S_ch = zeros(d,d,n,n);
  for i=1:n   
    for j=1:n
      [gmm_c_pre.covs(:,:,i,j),H_t(:,:,i,j),inv_S_ch(:,:,i,j),gmm_c_pre.denom(i,j)] = ...
        gaussian_multiply_special_precomp(gmm.covs(:,:,i),gmm.covs(:,:,j));
    end
  end
  gmm_c_pre.covs = reshape(gmm_c_pre.covs,d,d,n^2);
  gmm_c_pre.H = reshape(H_t,d,d*n^2)';
  gmm_c_pre.inv_S_ch_t = reshape(inv_S_ch,d,d*n^2)';
  gmm_c_pre.denom = reshape(gmm_c_pre.denom,n^2,1);
end

gmm_c.covs = gmm_c_pre.covs;
gmm_c.means = multiprod(gmm_c_pre.H,Z);
vc = multiprod(gmm_c_pre.inv_S_ch_t,Z);
numer = -0.5*sum(vc.*vc)';
log_c_c = numer - gmm_c_pre.denom;
log_mixweights = bsxfun(@plus,log(gmm.mixweights),log(gmm.mixweights)');
log_mixweights = log_mixweights(:) + log_c_c(:);
log_mixweights = log_mixweights - log_sum_exp(log_mixweights);
gmm_c.mixweights = exp(log_mixweights);


function [cov_c,H_t,inv_S_ch,denom] = gaussian_multiply_special_precomp(cov_a,cov_b)

S = cov_a + cov_b;
S_ch = chol(S);
inv_S_ch = inv(S_ch);
H_t = (cov_a*(inv_S_ch*inv_S_ch'))';
Wc = cov_a/S_ch;
cov_c = cov_a - Wc*Wc';
D = size(cov_a,1);
denom = 0.5*D*log(2*pi) + sum(log(diag(S_ch)));
