function [Dis, F, W, obj, res, runtime] = CGL(X, lambda, rho, Y)
tic;
k = 15;
V = length(X);
N = size(X{1}, 1);
c = max(Y);
S = cell(V, 1);
% data prepare
[norX] = NormalizeData(X);
for v = 1 : V
    S{v} = constructW_PKN(norX{v}', k);
end
L = cell(V, 1);
H = cell(V, 1);
HH = cell(V, 1);
hatH = cell(V, 1);
hatHH = cell(V, 1);
Q = cell(V, 1);
Z = cell(V, 1);
for v = 1:V
    DN = diag( 1./sqrt(sum(S{v})+eps) );
    L{v} = DN * S{v} * DN;
    H{v} = zeros(N, c);
    HH{v} = H{v}*H{v}';
    Q{v} = zeros(N, N);
    hatH{v} = Q{v}*H{v};
    hatHH{v} = hatH{v}*hatH{v}';
    Z{v} = zeros(N, N);
end
f = zeros(V, 1);
for v = 1:V
    f(v) = 0.5*lambda*norm(L{v} - HH{v}, 'fro')^2;
end
obj(1) = sum(f);
max_iter = 100;
% loop
for iter = 2 : max_iter
    % update H
    temp = cell(V, 1);
    G = cell(V, 1);
    for v = 1 : V
        temp{v} = Q{v} * ((0.5 * (Z{v} + Z{v}') - 0.5 * hatHH{v} )) * Q{v};
        G{v} = lambda * L{v} + temp{v};
        [H{v}] = eig2(G{v}, c);
        HH{v} = H{v} * H{v}';
        Q{v} = diag(1 ./ sqrt(diag(HH{v})));
        hatH{v} = Q{v} * H{v};
        hatHH{v} = hatH{v} * hatH{v}';
    end
    hatHH_tensor = cat(3, hatHH{ : , : });
    % update Z_tensor
    temphatHH = shiftdim(hatHH_tensor, 1);
    [tempZ, wtnn, trank] = prox_wtnn(temphatHH, rho);
    Z_tensor = shiftdim(tempZ, 2);
    for v = 1 : v
        Z{v} = Z_tensor( : , : , v);
    end
    % update obj
    f = zeros(V, 1);
    for v = 1 : V
        f(v) = 0.5*lambda * norm(L{v} - HH{v}, 'fro')^2 + 0.5 * norm(Z{v} - hatHH{v}, 'fro') ^ 2;
    end
    obj(iter) = sum(f) + wtnn;
    % convergence
    if iter > 40 && abs((obj(iter) - obj(iter - 1)) / obj(iter - 1)) < 1e-6
        break;
    end
end
Dis = zeros(N, N);
for v = 1 : V
    Dis = Dis + hatHH{v};
end
Dis = Dis / V;
disp(['iter:', num2str(iter)]);
time1 = toc;
tic;
W = constructW_PKN2(2 - Dis, k);
[~, F, ~, ~] = SpectralClustering(W, c);
[res] = myNMIACC(real(F), Y, c);
time2 = toc;
runtime = time1 + time2/20;


