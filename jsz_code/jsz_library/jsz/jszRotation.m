function [K0Q_cP, K1Q_cP, rho0_cP, rho1_cP] = jszRotation(W, K1Q_X, K0Q_X, rho1_X, rho0_X, BX, AX)
% function [K0Q_cP, K1Q_cP, rho0_cP, rho1_cP] = jszRotation(W, K1Q_X, K0Q_X, rho1_X, rho0_X, BX, AX)
%
%
% Inputs:
%   W          : N*J,      vector of portfolio weights to fit without error.
%   K1Q_X      : N*N
%   K0Q_X      : N*1
%   rho1_X     : N*1
%   rho0_X     : 1*1 
%   BX         : N*J  
%   AX         : 1*J
%
% Returns:
%   K0Q_cP : N*1
%   K1Q_cP : N*N
%   rho0_cP : scalar
%   rho1_cP : N*1
%
%
% r(t) = rho0_cP + rho1_cP'*cPt
%      = 1'*Xt
%      = 1 period discount rate (annualized)
%
% Under Q:
%   X(t+1) - X(t)   = K0Q_C + K1Q_X*X(t)  + eps_X(t+1),   cov(eps_X(t+1)) = Sigma_X
%   cP(t+1) - cP(t) = K0Q_cP + K1Q_cP*X(t) + eps_cP(t+1),  cov(eps_cP(t+1)) = Sigma_cP
%
% Where Sigma_X is chosen to match Sigma_cP 
%
% cPt = W*yt  (cPt N*1, W is N*J)
%     = W*AX' + W*BX'*Xt
%     = WAXp + WBXp*Xt
%
% Delta(cP) = WBXp*Delta(Xt)
%           = WBXp*(K0_X + K1Q_X*Xt + sqrt(Sigma_X)*eps(t+1))
%           = WBXp*K0Q_X + WBXp*(K1Q_X)*(WBXp\(cPt - WAXp)) + sqrt(Sigma_cP)*eps(t+1)
%           = WBXp*(K1Q_X)/WBXp*cPt +[WBXp*K0Q_X - WBXp*(K1Q_X)/WBXp*WAXp] + sqrt(Sigma_cP)*eps(t+1)
%
% rt = rho0_X + rho1_X'*Xt  [annualized 1-period rate]
%    = rho0_X + rho1_X'*(WBXp\(cPt - WAXp))
%    = [rho0_X - rho1_X'*(WBXp\WAXp)] + ((WBXp)'\rho1_X)'*cPt


WBXp = W*BX';
WAXp = W*AX';

K1Q_cP = WBXp*K1Q_X/WBXp;
K0Q_cP = WBXp*K0Q_X - K1Q_cP*WAXp;

rho0_cP = rho0_X - rho1_X'*(WBXp\WAXp);
rho1_cP = (WBXp)'\rho1_X;
