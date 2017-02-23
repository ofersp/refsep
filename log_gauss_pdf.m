% Evaluate log-gaussian PDF.
%
% log_prob = log_gauss_pdf(X,sigma,mu)
function log_prob = log_gauss_pdf(X,sigma,mu)

d = size(X,1);

if exist('mu','var')
  X = bsxfun(@minus,X,mu);
end

[R,p]= chol(sigma);
if p ~= 0
  error('sigma is not positive definite'); 
end

Q = R'\X;
q = sum(Q.^2); % quadratic term (M distance)
c = d*log(2*pi)+2*sum(log(diag(R))); % normalization constant
log_prob = -(c+q)/2;
