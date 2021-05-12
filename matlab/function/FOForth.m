function [X,Out] = FOForth(X, G, fun, opts, varargin)
% using a gradient reflection (or projection) method to solve
%
%       min             f(X):= E(X)+trace(G'X)
%       s. t.         X'X = I,  where X in R^{n*p}.
%
%  Assume gradient of E(X)=H(X)X, where H(X) is a  n-by-n symmetric matrix.
% ----------------------------------
%  Input:
%                           X --- n-by-p initial point such that X'X=I
%                           G --- n-by-p matrix
%                       fun --- a matlab function for f(X)
%                                  call: [funX,F] = fun(X,data1,data2);
%                                  funX: function value f(X)
%                                        F:  gradient of f(X)
%                                 data: extra data (can be more)
%             varargin --- data1, data2
%
%  Calling syntax:
%                     [X, out]= FOForth(X0,G,@fun,opts,data1,data2);
%
%                      opts --- option structure with fields:
%                                  solver:  1(gradient reflection method) 2(projection method) 3(QR retraction)
%                             stepsize:  0(ABB stepsize) o.w.(fixed stepsize)
%                                      xtol:  stop control for ||X_k - X_{k+1}||/sqr(n)
%                                      gtol:  stop control for ||kkt||/||kkt0||
%                                      ftol:  stop control for ||f_k -f_{k+1}||/(|f_k|+ 1)
%                                    maxit:  max iteration
%                                      info:  0(no print) o.w.(print)
%
%  Output:
%                            X --- solution
%                        Out --- output information
%                                        kkt: ||kkt|| (first-order optimality condition)
%                                      fval:  function value of solution
%                                      feaX: ||I-X'X||_F (feasiblity violation)
%                                      xerr: ||X_k - X_{k+1}||/sqr(n)
%                                      iter: total iteration number
%                                    fvals: history of function value
%                                      kkts: history of kkt
%                                message: convergence message
% --------------------------------------------------------------------
%  Reference:
%   B. Gao, X. Liu, X. Chen and Y. Yuan
%   A new first-order algorithmic framework for optimization problems with
%   orthogonality constraints, SIAM Journal on Optimization, 28 (2018), pp.302--332.
% ----------------------------------
%  Author: Bin Gao, Xin Liu (ICMSEC, AMSS, CAS)
%                 gaobin@lsec.cc.ac.cn
%                 liuxin@lsec.cc.ac.cn
%  Version: 1.0 --- 2016/04/01
%  Version: 1.1 --- 2017/10/16: support general function
% --------------------------------------------------------------------
%% default setting
if nargin < 4;opts=[];end

if isempty(X)
    error('Input X is an empty matrix');
else
    [n, p] = size(X);
end

if isempty(G);G = zeros(n,p);end

if isfield(opts, 'solver')
    if all(opts.solver ~= 1:3)
        opts.solver = 1;
    end
else
    opts.solver = 1;
end

if isfield(opts, 'stepsize')
    if opts.stepsize < 0
        opts.stepsize = 0;
    end
else
    opts.stepsize = 0;
end

if isfield(opts, 'xtol')
    if opts.xtol < 0 || opts.xtol > 1
        opts.xtol = 1e-8;
    end
else
    opts.xtol = 1e-8;
end

if isfield(opts, 'gtol')
    if opts.gtol < 0 || opts.gtol > 1
        opts.gtol = 1e-5;
    end
else
    opts.gtol = 1e-5;
end

if isfield(opts, 'ftol')
    if opts.ftol < 0 || opts.ftol > 1
        opts.ftol = 1e-10;
    end
else
    opts.ftol = 1e-10;
end

if isfield(opts, 'maxit')
    if opts.maxit < 0 || opts.maxit > 10000
        opts.maxit = 1000;
    end
else
    opts.maxit = 1000;
end

if ~isfield(opts, 'info');opts.info = 0;end
%% ---------------------------------------------------------------
% copy parameters
solver = opts.solver;
stepsize = opts.stepsize;
xtol = opts.xtol;
gtol = opts.gtol;
ftol = opts.ftol;
maxit = opts.maxit;
info = opts.info;

global Ip
Ip = eye(p);

% successive infomation (successive T iterations)
T = 5;  Terr = zeros(T,2);

%% ---------------------------------------------------------------
% Initialization
iter = 0; Out.fvals = []; Out.kkts = [];
% ensure X is orthogonal
if norm(X'*X-Ip,'fro')>1e-13; [X,~] = qr(X,0); end
% evaluate function and gradient info.
[funX, F] = feval(fun, X , varargin{:});
[PF,kkt0,feaX] = getPG(X,F);
% save history
Out.fvals(1) = funX; Out.kkts(1) = kkt0;

% initial stepsize
if stepsize == 0
    tau = max(0.1,min(0.01*kkt0,1));
else
    tau = stepsize;
end

% initial solver
switch solver
    case 1; mainsolver = @gradient_reflection;
    case 2; mainsolver = @projection;
    case 3; mainsolver = @QRretraction;
end

% info
if info ~= 0
    switch solver
        case 1
            fprintf('------------------ FOForth with gradient reflection start ------------------\n');
        case 2
            fprintf('------------------ FOForth with projection start ------------------\n');
        case 3
            fprintf('------------------ Riemannian Opt with QR retraction start ------------------\n');
    end
    fprintf('%4s | %15s | %10s | %10s | %8s | %8s\n', 'Iter ', 'F(X) ', 'KKT ', 'Xerr ', 'Feasi ', 'tau');
    fprintf('%d \t %f \t %3.2e \t %3.2e \t %3.2e \t %3.2e\n',iter, funX, kkt0, 0, feaX, tau);
end

%% ---------------------------------------------------------------
% main iteration
for iter = 1:maxit
    Xk = X;     Fk = F;    PFk = PF;   funXk = funX;
    
    % ---------- PART I: gradient step ----------
    %  Riemmanian or Euclidean gradient
    if solver == 3
        Grad = PFk;
    else
        Grad = Fk;
    end
    
    % gradient step
    V = Xk - tau * Grad;
    X = mainsolver(Xk,V);
    
    % ---------- PART II: symmetrization step ----------
    if feaX>1e-12; [X,~] = qr(X,0); end
    if solver ~= 3
        if G ~= 0
            [tu,~,tv] = svd(X'*G,0);
            X = -X*(tu*tv');
        end
    end
    
    % ------------ evaluate error ------------
    [funX, F] = feval(fun, X , varargin{:});
    [PF,kkt,feaX] = getPG(X,F);
    Out.fvals(iter+1) = funX;   Out.kkts(iter+1) = kkt;
    
    xerr = norm(Xk - X,'fro')/sqrt(n);
    ferr = abs(funXk - funX)/(abs(funXk)+1);
    
    % successive error
    Terr(2:T,:) =  Terr(1:(T-1),:); Terr(1,:) = [xerr, ferr]';
    merr = mean(Terr(1:min(iter,T),:));
    % info
    if info ~= 0 && (mod(iter,15) == 0 )
        fprintf('%d \t %f \t %3.2e \t %3.2e \t %3.2e \t %3.2e\n',iter, funX, kkt, xerr, feaX, tau);
    end
    
    % ------------ update ABB stepsize ------------
    if stepsize == 0
        Sk = X - Xk;
        Vk = PF - PFk;     %Vk = F - Fk;
        SV = sum(sum(Sk.*Vk));
        if mod(iter+1,2) == 0
            tau = abs(SV)/sum(sum(Vk.*Vk)); % SBB for odd
        else
            tau = sum(sum(Sk.*Sk))/abs(SV);  % LBB for even
        end
        tau = max(min(tau, 1e10), 1e-10);
    end
    
    % ------------------ stop criteria --------------------
    %     if kkt/kkt0 < gtol
    %     if kkt/kkt0 < gtol || (xerr < xtol || ferr < ftol)
    %     if kkt/kkt0 < gtol || (xerr < xtol && ferr < ftol)
    %     if (kkt/kkt0 < gtol && xerr < xtol) || ferr < ftol
    %     if (kkt/kkt0 < gtol && ferr < ftol) || xerr < xtol
    if kkt/kkt0 < gtol || (xerr < xtol && ferr < ftol) || all(merr < 10*[xtol, ftol])
        Out.message = 'converge';
        break;
    end
end

if iter >= opts.maxit
    Out.message = 'exceed max iteration';
end

Out.feaX = feaX;
Out.fval = funX;
Out.iter = iter;
Out.xerr = xerr;
Out.kkt = kkt;

if info ~= 0
    fprintf('%s at...\n',Out.message);
    fprintf('%d \t %f \t %3.2e \t %3.2e \t %3.2e \t %3.2e\n',iter, funX, kkt, xerr, feaX, tau);
    fprintf('------------------------------------------------------------------------\n');
end

%% ---------------------------------------------------------------
% nest-function
% gradient reflection step
    function X = gradient_reflection(X,V)
        VV = V'*V;
        VX = V'*X;
        TVX = VV\VX;
        X = -X + V*(2*TVX);
    end

% projection step
    function X = projection(~,V)
        % approach 1
        VV = V'*V;
        [Q,D] = eig(VV);
        DD = sqrt(D)\Ip;
        X = V*(Q*DD*Q');
        
        % approach 2
        %  [UX,~,VX] = svd(V,0);
        %   X = UX*VX';
    end

% QR retraction
    function X = QRretraction(~,V)
        % approach 1
        VV = V'*V;
        L = chol(VV,'lower');
        X = V*inv(L)';
        
        % approach 2
        %  [X,~] = qr(V,0);
    end

% get projected gradient and its norm
    function [PF,normPF,feaX] = getPG(X,F)
        PF = F - X*(F'*X);
        normPF = norm(PF,'fro');
        feaX = norm(X'*X-Ip,'fro');
    end

end