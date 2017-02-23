%
% C = multiprod(A,B)
function C = multiprod(A,B)

if numel(size(A)) == 3
  
  n = size(A,1);
  m = size(A,2);
  k = size(A,3);  
  assert(size(B,1) == m);
  assert(size(B,2) == k);
  A = reshape(A,[n,m*k]);
  B = repmat(B(:)',[m,1]);
  AB = A.*B;
  rAB = reshape(AB,[n,m,k]);
  sumAB = sum(rAB,2);
  C = reshape(sumAB,[m,k]);

elseif numel(size(A)) == 2
  
  n = size(A,1);
  m = size(A,2);
  k = n/m;
  C = reshape(A*B,m,k);
end
  
  
