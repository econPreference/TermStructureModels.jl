function [Gamma_hat, alpha_hat, Omega_hat] = regressVAR(X)
% function [Gamma_hat, alpha_hat, Omega_hat] = regressVAR(X)
% X: T*N
%
% Gamma_hat : N*N
% alpha_hat : N*1
% Omega_hat : N*N
%
% X(t+1) = alpha + Gamma*X(t) + eps(t+1), cov(eps(t+1)) = Omega
%
% Compute the maximum likelihood estimates of Gamma, alpha, Omega
%
% NOTE: The MLE estimates of Gamma, alpha do not depend on Omega.
% That is, the argmax_{Gamma,alpha} [L(X|Gamma,alpha,Omega)] = f(X)
% So this function provides MLE of Gamma, alpha for a fixed Omega.

[T,N] = size(X);

Yt = X(1:end-1,:);  % (T-1)*N
Ytp1 = X(2:end,:);  % (T-1)*N

Y = Ytp1.';  % N*(T-1) 
Z = [ones(T-1,1), Yt].'; % (N+1)*(T-1)
A = Y*Z.'*inv(Z*Z.'); % N*(N+1)
alpha_hat = A(:,1);
Gamma_hat = A(:,2:end);

if nargout==3
    residuals = Ytp1 - (A*Z).'; % (T-1)*N
    Omega_hat = 1/(T-1)*residuals.'*residuals;
end

