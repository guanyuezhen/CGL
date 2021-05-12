function [H1,H2, LapN, D] = SpectralClustering(CKSym, c)
warning off;
% Normalized spectral clustering according to Ng & Jordan & Weiss
% using Normalized Symmetric Laplacian L = D^{-1/2} W D^{-1/2}
DN = diag( 1./sqrt(sum(CKSym)+eps) );
LapN = DN * CKSym * DN;
[H1] = eig2(LapN, c);
D = diag( 1./sqrt(diag(H1*H1')) );
H2 = H1 ./ sqrt(sum(H1.^2, 2));
if any(isnan(H2))
    H2(isnan(H2)) = 0;
end
H2 = real(H2);
end