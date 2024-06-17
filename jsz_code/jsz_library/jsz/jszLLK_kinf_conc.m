function [llk, AcP, BcP, AX, BX, kinf, K0P_cP, K1P_cP, sigma_e, K0Q_cP, K1Q_cP, rho0_cP, rho1_cP, cP, llkP, llkQ, K0Q_X, K1Q_X, rho0_X, rho1_X] = ...
    jszLLK_kinf_conc(yields_o, W, K1Q_X, Sigma_cP, mats, dt, K0P_cP, K1P_cP, sigma_e, rankRP)
%function [llk, AcP, BcP, AX, BX, K0P_cP, K1P_cP, sigma_e, K0Q_cP, K1Q_cP, rho0_cP, rho1_cP, cP, llkP, llkQ] = ...
%    jszLLK_kinf(yields_o, W, yields_woe, K1Q_X, rinfQ, Sigma_cP, mats, dt, K0P_cP, K1P_cP, sigma_e, rankRP)
%
%
% This function computes the likelihood for a Gaussian term structure.
% See "A New Perspective on Gaussian Dynamic Term Structure Models" by Joslin, Singleton and Zhu
%
% INPUTS:
% yields_o   : (T+1)*J,  matrix of observed yields (first row are t=0 observations, which likelihood conditions on)
% mats       : 1*J,      maturities in years
% dt         : scalar,   length of period in years
%
% W          : N*J,      vector of portfolio weights to fit without error.
% K1Q_X      : N*N,      normalized latent-model matrix (does not have to be diagonal, see form below)
% Sigma_cP   : N*N,      positive definite matrix that is the covariance of innovations to cP
%
% OPTIONAL INPUTS:
% K0P_cP     : N*1,      OPTIONAL (supply [] to omit)
% K1P_cP     : N*N,      OPTIONAL (supply [] to omit)
% sigma_e    : scalar,   standard error of yield observation errors
%
%
% Compute likelihood conditioned on first observation!
%
% llk        : T*1       time series of -log likelihoods (includes 2-pi constants)
% AcP        : 1*J       yt = AcP' + BcP'*Xt  (yt is J*1 vector)
% BcP        : N*J       AcP, BcP satisfy internal consistency condition that AcP*W' = 0, BcP*W' = I_N
% AX         : 1*J       yt = AX' + BX'*Xt  
% BX         : N*J       Xt is the 'jordan-normalized' latent state
%
%
% The model takes the form:
%   r(t) = rho0_cP + rho1_cP'*cPt
%        = rinfQ + 1'*Xt  (Xt is the 'jordan-normalized' state
%        = 1 period discount rate (annualized)
%
% Under Q:
%   X(t+1) - X(t)   = K0Q_X  + K1Q_X*X(t)  + eps_X(t+1),   cov(eps_X(t+1)) = Sigma_X
%   cP(t+1) - cP(t) = K0Q_cP + K1Q_cP*X(t) + eps_cP(t+1),  cov(eps_cP(t+1)) = Sigma_cP
%   where Sigma_X is chosen to match Sigma_cP 
% and K0Q_X(m1) = kinfQ where m1 is the multiplicity of the highest eigenvalue (typically 1)
%
% Under P:
%   cP(t+1) - cP(t) = K0P_cP + K1P_cP*X(t) + eps_cP(t+1),  cov(eps_cP(t+1)) = Sigma_cP
%
% Model yields are given by:
%   yt^m = AcP' + BcP'*cPt  (J*1)
% And observed yields are given by:
%  yt^o = yt^m + epsilon_e(t)
% where V*epsilon_e~N(0,sigma_e^2 I_(J-N))
% and V is an (J-N)*J matrix which projects onto the span orthogonal to the
% row span of W.  This means errors are orthogonal to cPt and cPt^o = cPt^m.
%
    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup 
[T,J] = size(yields_o(2:end,:));
% Setup W if we are using individual yields without error:

N = size(W,1);
cP = yields_o*W'; % (T+1)*N, cP stands for math caligraphic P.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup default parameters:
concentrateSigmaE = true;
concentrateK0PK1P = true;
if exist('K0P_cP','var') && ~isempty(K0P_cP), concentrateK0PK1P = false; end
if ~exist('rankRP','var') || isempty(rankRP), rankRP = N; end
if exist('sigma_e','var') && ~isempty(sigma_e), concentrateSigmaE = false; end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% COMPUTE THE Q-LIKELIHOOD:
% First find the loadings for the model:
% yt = AcP' + BcP'*cPt, AcP is 1*J, BcP is N*J
% AcP = alpha0_cP*kinf + alpha1_cP

rho0_cP = 0;
% AcP0, AX0 will be the loadings with rho0_cP = 0, which won't be correct
[BcP, AcP0, K0Q_cPx, K1Q_cP, rho0_cP, rho1_cP, K0Q_X, K1Q_X, AX0, BX, Sigma_X, alpha0_cP, alpha1_cP, alpha0_X, alpha1_X, m1] = jszLoadings_rho0cP(W, K1Q_X, rho0_cP, Sigma_cP, mats, dt);
V = null(W)';
kinf = ((mean(yields_o(2:end,:))' - alpha1_cP' - BcP'*mean(cP(2:end,:)).')'*(V'*V*alpha0_cP')) / (alpha0_cP*V'*V*alpha0_cP');
AX = alpha0_X*kinf + alpha1_X;
AcP = alpha0_cP*kinf + alpha1_cP;
K0Q_X = zeros(N,1);
K0Q_X(m1) = kinf;

rho0_X = 0;
rho1_X = ones(N,1);
[K0Q_cP, K1Q_cP, rho0_cP, rho1_cP] = jszRotation(W, K1Q_X, K0Q_X, rho1_X, rho0_X, BX, AX);

yields_m =  ones(T+1,1)*AcP + cP*BcP; % (T+1)*J, model-implied yields
yield_errors = yields_o(2:end,:) - yields_m(2:end,:); % T*J
square_orthogonal_yield_errors = (yield_errors).^2; % T*J, but N-dimensional projection onto W is always 0, so effectively (J-N) dimensional

% Compute optimal sigma_e if it is not supplied
if concentrateSigmaE
    sigma_e = sqrt( sum(square_orthogonal_yield_errors(:))/(T*(J-N)) );
end

llkQ = .5*sum(square_orthogonal_yield_errors.')/sigma_e^2 + (J-N)*.5*log(2*pi) + .5*(J-N)*log(sigma_e^2); % 1*T
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% COMPUTE THE P-LIKELIHOOD:
% We have three case: (1) unconstrained, (2) reduced rank risk premia and (3) kalman filter
if concentrateK0PK1P 
    if rankRP==N
        % Run OLS to obtain maximum likelihood estimates of K0P, K1P
        [K1PplusI, K0P_cP] = regressVAR(cP);
        K1P_cP = K1PplusI - eye(N);
    else
        % Run reduced rank regression to compute ml estimates of K0P, K1P
        Y1 = diff(cP) - cP(1:end-1,:)*K1Q_cP.';
        X = cP(1:end-1,:);
        [beta1, alpha1] = reducedRankFreeInterceptRegress(Y1, X, rankRP, Sigma_cP);
        K1P_cP = beta1.' + K1Q_cP;
        K0P_cP = alpha1.';
% THIS WOULD DO REDUCED RANK of [K0P, K1P] - [K0Q, K1Q] (only rankRP are priced instead of only rankRP have time-varying price)       
%        Y1 = diff(cP) - cP(1:end-1,:)*K1Q_cP.' - ones(T,1)*K0Q_cP.';
%        X = [ones(T,1), cP(1:end-1,:)];
%        K1P_cP = beta1(2:end,:).' + K1Q_cP;
%        K0P_cP = beta1(1,:).' + K0Q_cP;
    end
end
innovations = cP(2:end,:).' - (K0P_cP*ones(1,T) + (eye(N)+K1P_cP)*cP(1:end-1,:).'); % N*T
llkP = .5*N*log(2*pi) + .5*log(det(Sigma_cP)) + .5*sum(innovations.*(Sigma_cP\innovations),1); % 1*T
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




llk = (llkQ + llkP).'; % T*1 series

