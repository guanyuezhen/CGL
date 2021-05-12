function [F] = eig2(G, c)
N = size(G, 1);
if N < 10000 % using cpu
    opt.disp = 0;
    [F, ~] = eigs(G, c, 'la', opt);
else % using gpu
    tempG = gpuArray(single(G));
    [v, d] = eig(tempG);
    d = diag(d);
    [~, idx] = sort(d,'descend');
    idx1 = idx(1:c);
    tempF = v(:,idx1);
    F= double(gather (tempF)); 
end
