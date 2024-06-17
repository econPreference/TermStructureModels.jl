function [By,Ay, dAyH0d] = gaussianDiscreteYieldLoadingsDiagonal(maturities, K0d, K1d_diag, H0d, rho0d, rho1d, timestep)
%function [By,Ay, dAyH0d] = gaussianDiscreteYieldLoadingsDiagonal(maturities, K0d, K1d_diag, H0d, rho0d, rho1d, timestep)
%
% DOESN'T HANDLE UNIT ROOTS!!
%
% THIS FUNCTION ASSUMES K1d is diagonal
% K0d      : N*1
% K1d_diag : N*1
% H0d      : N*N
% rho0d    : scalar  
% rho1d    : N*1
% timestep : optional argument.
%
% By : N*M
% Ay : 1*M  (faster to not compute with only one output argument)
%
% r(t) = rho0d + rho1d'Xt
%      = 1 period discount rate
% P(t) =  price of  t-period zero coupon bond
%      = EQ0[exp(-r0 - r1 - ... - r(t-1)]
%      = exp(A+B'X0)
% yields = Ay + By'*X0
%   yield is express on a per period basis unless timestep is provided.
%   --For example, if the price of a two-year zero is exp(-2*.06)=exp(-24*.005),
%   --and we have a monthly model, the function will return Ay+By*X0=.005
%   --unless timestep=1/12 is provided in which case it returns Ay+By*X0=.06
%
% Where under Q:
%   X(t+1) - X(t) = K0d + K1d*X(t) + eps(t+1),  cov(eps(t+1)) = H0d
%
% We can compute the loadings by recurrence relations:
%   A1 = -rho0d
%   B1 = -rho1d
%   At = A(t-1) + K0d'*B(t-1) .5*B(t-1)'*H0d*B(t-1) - rho0d
%   Bt = B(t-1) + K1d'*B(t-1) - rho1d
%
% Or in closed form by noting that 
%    r0+r1+..+r(t-1) = c.X(0) + alpha0 + alpha1*eps1 + ... + alpha(t-1)*eps(t-1)
%                    ~ N(c.X(0) + alpha0, alpha1'H0d*alph1 + ... + alpha(t-1)'*H0d*alpha(t-1))
%
% And then use the MGF of Y~N(mu,Sigma) is E[exp(a.Y)] = a'*mu + .5*a'*Sigma*a
% (or similarly use the partial geometric sum formulas repeatedly)
%
% Let G = K1+I
% X(0)
% X(1) = K0 + G*X(0) + eps1
% X(2) = K0 + G*K0 + G^2*X(0) + G*eps1 + eps2
% X(3) = K0 + G*K0 + G^2*K0 + G^3*X(0) + G^2*eps1 + G*eps2 + eps3
% X(n) = sum(I+G+..+G^(n-1))*K0 + G^n*X(0) + sum(i=1..n,G^(n-i)*epsi)
%      = (I-G\(I-G^n)*K0 + G^n*X(0) + sum(i=1..n,G^(n-i)*epsi)
%
% cov(G^n*eps) = G^n*cov(eps)*(G^n)'
% vec(cov(G^n*eps) = kron(G^n,G^n)*vec(eps)
%                   = (kron(G,G)^n)*vec(eps)
%
% sum(X(i),i=1..n) = mu0 + mu1*X0 + u
%    mu0 = (I-G)\(I - (I-G)\(G-G^(n+1))*K0 
%    mu1 = (I-G)\(G-G^(n+1))
%    vec(cov(u)) = see below.  
%  u = (I-G)\(I-G^n)*eps1 + 
%      (I-G)\(I-G^(n-1))*eps2 + ..
%      (I-G)\(I-G)*epsn
%  cov(u) = (I-G)\Sig/(I-G)'
% Sig = sum(cov(eps)) + sum(i=1..n,G^i*cov(eps)) + 
%       sum(i=1..n,cov(eps)G^i') + sum(i=1..n,G^i*cov(eps)*G^i')
% compute the last one using vec's.  see below.


% K1d_diag is N*1 -- the diagonal of K1d
M = length(maturities);
N = length(K0d);
Ay = zeros(1,M);
By = zeros(N, M);
dAyH0d = zeros(N,N,M);

I = ones(N,1);
G = K1d_diag+ones(N,1); % N*1
if nargout>1
    GG = G*G'; % N*N
    GpG = G*ones(1,N) + ones(N,1)*G'; % N*N (i,j) entry is G(i)+G(j)
end

for m=1:M
    mat = maturities(m);
    if mat==1
        By(:,m) = rho1d;
        Ay(:,m) = rho0d;                
        mu0 = zeros(N,1);
        mu1 = ones(N,1);
        Sigma0 = zeros(N);
        continue
    end
    
    i = mat-1;  % # of random innovations X(0) + X(1) + ... + X(mat)=X(0)+...+X(i)
    % X(0) + ... + X(i)~N(mu0 + mu1*X(0), Sigma0)
    mu1 = I  - K1d_diag.\(G - G.^(i+1)); % N*1
    mu0 = -K1d_diag.\((i+1)*I - mu1).*K0d;
    By(:,m) = rho1d.*mu1/mat;
    
    if nargout>1
        Sigma_term1 = i*H0d; % N*N
        Sigma_term2 = ((mu1 - 1)*ones(1,N)).*H0d; % N*N
        Sigma_term3 = Sigma_term2.';
        Sigma_term4 = (1 - GG(:)).\(GG(:) - GG(:).^(i+1)).*H0d(:);
        Sigma_term4 = reshape(Sigma_term4, [N,N]);
        Sigma0 = (K1d_diag*K1d_diag.').\(Sigma_term1 - Sigma_term2 - Sigma_term3 + Sigma_term4);

        Ay(:,m) = rho0d + (rho1d.'*mu0 - .5*rho1d.'*Sigma0*rho1d)/mat;
    end
    if nargout>2
        dAyH0d(:,:,m) = -.5*i*rho1d*rho1d' ...
            +.5*((mu1-1)*ones(1,N)).*(rho1d*rho1d.') ...
            +.5*(ones(N,1)*(mu1-1).').*(rho1d*rho1d.') ...
            - .5*reshape((1 - GG(:)).\(GG(:) - GG(:).^(i+1)), [N,N]).*(rho1d*rho1d.');
        dAyH0d(:,:,m) = (K1d_diag*K1d_diag.').\dAyH0d(:,:,m)/mat;
    end
end

if nargin==7 
    By = By/timestep;
    if nargout>1
        Ay = Ay/timestep;
        if nargout>2
            dAyH0d = dAyH0d/timestep;
        end
    end            
end
