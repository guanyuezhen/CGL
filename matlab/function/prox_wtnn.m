  function [X, wtnn, trank] = prox_wtnn(Y, C)

% The proximal operator of the tensor nuclear norm of a 3 way tensor
%
% min_X ||X||_w*+0.5*||X-Y||_F^2
%
% Y     -    n1*n2*n3 tensor
%
% X     -    n1*n2*n3 tensor
% tnn   -    tensor nuclear norm of X
% trank -    tensor tubal rank of X
%
% version 2.1 - 14/06/2018
%
% Written by Canyi Lu (canyilu@gmail.com)
%
%
% References: 
% Canyi Lu, Tensor-Tensor Product Toolbox. Carnegie Mellon University. 
% June, 2018. https://github.com/canyilu/tproduct.
%
% Canyi Lu, Jiashi Feng, Yudong Chen, Wei Liu, Zhouchen Lin and Shuicheng
% Yan, Tensor Robust Principal Component Analysis with A New Tensor Nuclear
% Norm, arXiv preprint arXiv:1804.03728, 2018
%

[n1,n2,n3] = size(Y);
X = zeros(n1,n2,n3);
Y = fft(Y,[],3);
wtnn = 0;
trank = 0;
      
% first frontal slice
[U,S,V] = svd(Y(:,:,1),'econ');
S = diag(S);
temp = (S - eps).^2 - 4*(C - eps*S);
ind = find(temp>0);
r = length(ind);
if r>=1
    S = max(S(ind)-eps+sqrt(temp(ind)),0)/2;
    X(:,:,1) = U(:,1:r)*diag(S)*V(:,1:r)';
    wtnn = wtnn + sum(S.*(C./(S+eps)));
    trank = max(trank, r);
end
% i=2,...,halfn3
halfn3 = round(n3/2);
for i = 2 : halfn3
    [U,S,V] = svd(Y(:,:,i),'econ');
    S = diag(S);
    temp = (S - eps).^2 - 4*(C - eps*S);
    ind = find(temp>0);
    r = length(ind);
    if r>=1
        S = max(S(ind)-eps+sqrt(temp(ind)),0)/2;
        X(:,:,i) = U(:,1:r)*diag(S)*V(:,1:r)';
        wtnn = wtnn + sum(S.*(C./(S+eps)));
        trank = max(trank,r);
    end
    X(:,:,n3+2-i) = conj(X(:,:,i));
end

% if n3 is even
if mod(n3,2) == 0
    i = halfn3+1;
    [U,S,V] = svd(Y(:,:,i),'econ');
    S = diag(S);
    temp = (S - eps).^2 - 4*(C - eps*S);
    ind = find(temp>0);
    r = length(ind);
    if r>=1
        S = max(S(ind)-eps+sqrt(temp(ind)),0)/2;
        X(:,:,i) = U(:,1:r)*diag(S)*V(:,1:r)';
        wtnn = wtnn + sum(S.*(C./(S+eps)));
        trank = max(trank,r);
    end
end
wtnn = wtnn/n3;
X = ifft(X,[],3);