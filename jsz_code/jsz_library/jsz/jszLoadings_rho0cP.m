function [BcP, AcP, K0Q_cP, K1Q_cP, rho0_cP, rho1_cP, K0Q_X, K1Q_X, AX, BX, Sigma_X, alpha0_cP, alpha1_cP, alpha0_X, alpha1_X, m1] = jszLoadings_rho0cP(W, K1Q_X, rho0_cP, Sigma_cP, mats, dt, Sigma_X)
%function [BcP, AcP, K0Q_cP, K1Q_cP, rho0_cP, rho1_cP, K0Q_X, K1Q_X, AX, BX, Sigma_X, alpha0_cP, alpha1_cP, alpha0_X, alpha1_X, m1] = jszLoadings_rho0cP(W, K1Q_X, rho0_cP, Sigma_cP, mats, dt, Sigma_X)
%
%
% This gives a slight variant of the JSZ normalization.
% 
% There is one "intercept" parameter governing the risk-neutral
% distribution.  This can be the long run mean of the short rate under Q
% (assuming stationarity) or the drift of the most persistent factor.
%
% Here we parameterize the intercept parameter through the relationship:
%  r_t = rho0_cP + rho1_cP.Xt
% 
% Given a fixed set of eigenvalues for K1Q, there is a one-to-one
% mapping between rho0_cP and kinf.
%
%
% Inputs:
%   mats       : 1*J,      maturities in years
%   dt         : scalar,   length of period in years
%   W          : N*J,      vector of portfolio weights to fit without error.
%   K1Q_X      : N*N
%   rho0_cP    : scalar,   the short rate will be rt = rho0_cP + rho1_cP.cP_t
%   Sigma_cP, Sigma_X : N*N  covariance of innovations. PROVIDE ONE OR THE OTHER
%
% Returns:
%   AcP : 1*J
%   BcP : N*J
%   AX  : 1*J
%   BX  : N*J
%   Sigma_X : N*N
%   AcP = alpha0_cP*kinf + alpha1_cP
%   AX  = alpha0_X*kinf  + alpha1_X
%
%
% This function:
% 1. Compute the loadings for the normalized model:
%   X(t+1) - X(t)   = [kinfQ;0] + K1Q_X*X(t)  + eps_X(t+1),   cov(eps_X(t+1)) = Sigma_X
%     and r(t) = 1.X(t)  
%     where r(t) is the annualized short rate, (i.e. price of 1-period zero coupon bond at time t is exp(-r(t)*dt))
%    If Sigma_X is not provided, it is solved for so that Sigma_cP (below) is matched.
%    yt = AX' + BX'*Xt
%
% 2. For cPt = W*yt and the model above for Xt, find AcP, BcP so that
%    yt = AcP' + BcP'*cPt
%
%


J = length(mats);
N = size(K1Q_X,1);
rho0d = 0;
rho1d = ones(N,1);
mats_periods = round(mats/dt);
M = max(mats_periods);

[K1Q_X, isTypicalDiagonal, m1] = jszAdjustK1QX(K1Q_X);
K0Q_X = zeros(N,1);
K0Q_X(m1) = 1; 



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% First compute the loadings ignoring the convexity term -- BX will be correct
% yt = AX' + BX'*Xt
% yt is J*1
% AX is 1*J
% BX is N*J
% Xt is N*1
%
% cPt = W*yt  (cPt N*1, W is N*J)
%     = W*AX' + W*BX'*Xt
%     = WAXp + WBXp*Xt
%
% Substituting:
% yt = AX' + BX'*(WBXp\(cPt - WAXp))
%    = (I - BX'*(WBXp\WAXp))*AX' + BX'*WBXp\cPt
%    = AcP' + BcP'*cPt
% where AcP = AX*(I - BX'*(WBXp\WAXp))'
%       BcP = (WBXp)'\BX
%
% Sigma_cP = W*BX'*Sigma_X*(W*BX')'
% Sigma_X = (W*BX')\Sigma_cP/(W*BX')'
%


% If K1d isn't diagonal, we should use the Recurrence solver:.
% Since we are setting covariance to zero and K0Q_X = [1;0;0..;0], the "A" loadings will be the loading on kinf
if isTypicalDiagonal
    [BX, alpha0_X] = gaussianDiscreteYieldLoadingsDiagonal(mats_periods, K0Q_X, diag(K1Q_X), zeros(N,N), rho0d*dt, rho1d*dt, dt); % N*J
else
    [BX, alpha0_X] = gaussianDiscreteYieldLoadingsRecurrence(mats_periods, K0Q_X, K1Q_X, zeros(N,N), rho0d*dt, rho1d*dt, dt); % N*J
end

WBXp = W*BX.'; % N*N
if nargin<7 || isempty(Sigma_X)
    Sigma_X = (W*BX')\Sigma_cP/(BX*W');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now with Sigma_X in hand, compute loadings for AX
if isTypicalDiagonal
    [BX, AX1] = gaussianDiscreteYieldLoadingsDiagonal(mats_periods, K0Q_X, diag(K1Q_X), Sigma_X, rho0d*dt, rho1d*dt, dt);
else
    [BX, AX1] = gaussianDiscreteYieldLoadingsRecurrence(mats_periods, K0Q_X, K1Q_X, Sigma_X, rho0d*dt, rho1d*dt, dt);
end
% AX1 gives the intercept with K0Q_X all zeros except 1 in the m1-th entry.
% So AX = alpha0_X*kinf + alpha1_X which alpha1_X = AX1 - alpha0_X
alpha1_X = AX1 - alpha0_X;

WBXp = W*BX';

% Need to find what kinf should be to get the desired rho0_cP:
% rt = 1'*Xt 
% cPt = (W*alpha0_X')*kinf + (W*alpha1_X') + (W*BX')*Xt
% rt = 1'*(W*BX')^(-1)*[cPt -(W*alpha0_X')*kinf - (W*alpha1_X')]
% --> rho0_cP = -1'*(W*BX')^(-1)*(W*alpha0_X')*kinf - 1'*(W*BX')^(-1)*(W*alpha1_X')
a0 = ones(1,N)*(WBXp\(W*alpha0_X'));
a1 = ones(1,N)*(WBXp\(W*alpha1_X')); 
kinf = -(rho0_cP + a1)/a0;
K0Q_X(m1) = kinf; 

AX = alpha0_X*kinf + alpha1_X;
% AcP = alpha0_cP*kinf + alpha1_cP
alpha0_cP = ((eye(J) - (BX'/(W*BX'))*W)*alpha0_X')';
alpha1_cP = ((eye(J) - (BX'/(W*BX'))*W)*alpha1_X')';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Finally, rotate the model to obtain the AcP, BcP loadings.
% (See above for calculation)
BcP = (W*BX.').'\BX;
AcP = AX*(eye(J) - BX'*((W*BX')\W))'; % 1*J
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now compute the rotated model parameters:
WBXp = W*BX';
WAXp = W*AX';

K1Q_cP = WBXp*K1Q_X/WBXp;
K0Q_cP = WBXp*K0Q_X - K1Q_cP*WAXp;

rho0_cP =  - ones(1,N)*(WBXp\WAXp);
rho1_cP = (WBXp)'\ones(N,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%