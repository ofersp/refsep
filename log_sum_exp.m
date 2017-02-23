% Compute log(sum(exp(x),dim)) while avoiding numerical underflow.
%
% s = log_sum_exp(x, dim)
% By default dim = 1 (columns).
%
% Written by Michael Chen (sth4nth@gmail.com).
function s = log_sum_exp(x, dim)

if nargin == 1, 
  % Determine which dimension sum will use
  dim = find(size(x)~=1,1);
  if isempty(dim), dim = 1; end
end
% subtract the largest in each column
y = max(x,[],dim);
x = bsxfun(@minus,x,y);
s = y + log(sum(exp(x),dim));
i = find(~isfinite(y));
if ~isempty(i)
  s(i) = y(i);
end
