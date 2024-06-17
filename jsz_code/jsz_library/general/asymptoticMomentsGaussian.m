function [Minf, Vinf] = asymptoticMomentsGaussian(K0Pd, K1Pd, H0d)
% function [Minf, Vinf] = asymptoticMomentsGaussian(K0Pd, K1Pd, H0d)
%
%   X(t+1) - X(t) = K0Pd + K1Pd*X(t) + eps(t+1),  cov(eps(t)) = H0d
%
% Compute the stationary distribution of X is N(Minf,Vinf)
% (Assuming negative real parts of the eigenvalues of K1Pd)
%
% K0Pd : N*1
% K1Pd : N*N
% H0d  : N*N
% 
% Minf : N*1
% Vinf : N*N

N = length(K0Pd);

Minf = - K1Pd\K0Pd;
A = K1Pd + eye(N);
% vec(A*B*C) = kron(C'*A)*vec(B)
% Vinf = A*Vinf*A' + H0d
Vinf = reshape( (eye(N^2) - kron(A, A))\H0d(:),[N,N]);