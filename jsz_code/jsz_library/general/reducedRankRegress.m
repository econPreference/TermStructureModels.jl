function [beta, Omega, log_llk] = reducedRankRegress(Y, X, r, Omega)
%function [beta, Omega, log_llk] = reducedRankRegress(Y, X, r, Omega)
% 
% Y : T*N
% X : T*M
% r : scalar, rankd of beta
% Omega : N*N
% 
% beta  : M*N
% Omega : N*N
% log_llk : T*1 minus log of the likelihood (including 2pi's)
%
% Regress Y = X*beta + eps, cov(eps) = Omega
%
%   under the assumption that beta has rank r
% If Omega is not provided, likelihood is maximized over beta and Omega
%

[T N] = size(Y);
%M = size(X,2);

if nargin==2
    r = N;
end

if r==N
    beta = (X'*X)\(X'*Y); % M*N
else
    if nargin<4 || isempty(Omega)
        M00 = 1/(T) * Y'*Y;
        M0k = 1/(T) * Y'*X;
        Mkk = 1/(T) * X'*X;
        [V D] = eig(M0k'*(M00\M0k),Mkk);
        [A B] = sort(diag(D),'descend');
        b = V(:,B(1:r));
        %  b = b/b(1:r,1:r);
        a = M0k*b/(b'*Mkk*b);
        beta = (a*b')';
    else
        P = chol(X'*X);
        try
            L = chol(Omega);
        catch
            [U0 D0 V0] = svd(Omega);
            L = U0*sqrt(D0);
        end
        betaols = (X'*X)\(X'*Y);
        [U S V] = svd(P*betaols/L);
        S(r+1:end,r+1:end) = 0;
        beta = P\U*S*V'*L;
    end
end

if nargout>1 && (nargin<4 || isempty(Omega))
    %disp('compute omega')
    Omega = 1/T*(Y-X*beta).'*(Y-X*beta);
end

if nargout>2
    log_llk = +N/2*log(2*pi) + .5*log(det(Omega)) + .5*(sum((Y-X*beta).'.*(Omega\(Y-X*beta).'),1)).'; % T*1
end
