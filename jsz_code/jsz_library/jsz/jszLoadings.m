function [BcP, AcP, K0Q_cP, K1Q_cP, rho0_cP, rho1_cP, K0Q_X, K1Q_X, AX, BX, Sigma_X] = jszLoadings(W, K1Q_X, kinfQ, Sigma_cP, mats, dt, Sigma_X)
% function [BcP, AcP, K0Q_cP, K1Q_cP, rho0_cP, rho1_cP, K0Q_X, K1Q_X, AX, BX, Sigma_X] = jszLoadings(W, K1Q_X, kinfQ, Sigma_cP, mats, dt, Sigma_X)
%
% Inputs:
%   mats       : 1*J,      maturities in years
%   dt         : scalar,   length of period in years
%   W          : N*J,      vector of portfolio weights to fit without error.
%   K1Q_X      : N*N
%   kinfQ      : scalar,   when the model is stationary, the long run mean under Q is -kinfQ/K1 
%   Sigma_cP, Sigma_X : N*N  covariance of innovations. PROVIDE ONE OR THE OTHER
%
% Returns:
%   AcP : 1*J
%   BcP : N*J
%   AX  : 1*J
%   BX  : N*J
%   Sigma_X : N*N
%
%
% This function:
% 1. Compute the loadings for the normalized model:
%   X(t+1) - X(t)   = K0Q_X + K1Q_X*X(t)  + eps_X(t+1),   cov(eps_X(t+1)) = Sigma_X
%     and r(t) = 1.X(t)  
%     where r(t) is the annualized short rate, (i.e. price of 1-period zero coupon bond at time t is exp(-r(t)*dt))
%     and K0Q_X(m1) = kinf, and K0Q_X is 0 in all other entries.
%     m1 is the multiplicity of the first eigenvalue.
%
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
K0Q_X(m1) = kinfQ; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% If Sigma_cP is provided, we need to compute Sigma_X by 
% first computing BX
%
if nargin<7 || isempty(Sigma_X)
    % First compute the loadings ignoring the convexity term -- BX will be correct
    % yt = AX' + BX'*Xt  
    % yt is J*1, AX is 1*J
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
    if isTypicalDiagonal 
        BX = gaussianDiscreteYieldLoadingsDiagonal(mats_periods, K0Q_X, diag(K1Q_X), zeros(N,N), rho0d*dt, rho1d*dt, dt); % N*J
    else
        BX = gaussianDiscreteYieldLoadingsRecurrence(mats_periods, K0Q_X, K1Q_X, zeros(N,N), rho0d*dt, rho1d*dt, dt); % N*J       
    end

    WBXp = W*BX.'; % N*N
    Sigma_X = (W*BX')\Sigma_cP/(BX*W');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now with Sigma_X in hand, compute loadings for AX
if isTypicalDiagonal
    [BX, AX] = gaussianDiscreteYieldLoadingsDiagonal(mats_periods, K0Q_X, diag(K1Q_X), Sigma_X, rho0d*dt, rho1d*dt, dt);
else
    [BX, AX] = gaussianDiscreteYieldLoadingsRecurrence(mats_periods, K0Q_X, K1Q_X, Sigma_X, rho0d*dt, rho1d*dt, dt);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Rotate the model to obtain the AcP, BcP loadings.
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