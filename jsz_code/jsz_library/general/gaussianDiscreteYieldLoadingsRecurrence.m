function [By, Ay] = gaussianDiscreteYieldLoadingsRecurrence(maturities, K0d, K1d, H0d, rho0d, rho1d, timestep)
% function [By, Ay] = gaussianDiscreteYieldLoadingsRecurrence(maturities, K0d, K1d, H0d, rho0d, rho1d, timestep)
%
% K0d      : N*1
% K1d      : N*1
% H0d      : N*N
% rho0d    : scalar  
% rho1d    : N*1
% timestep : optional argument.
%
% By : N*M
% Ay : 1*M  (faster to not compute with only one output argument)
%
% r(t)   = rho0d + rho1d'Xt
%        = 1 period discount rate
% P(t)   =  price of  t-period zero coupon bond
%        = EQ0[exp(-r0 - r1 - ... - r(t-1)]
%        = exp(A+B'X0)
% yields = Ay + By'*X0
%   yield is express on a per period basis unless timestep is provided.
%   --For example, if the price of a two-year zero is exp(-2*.06)=exp(-24*.005),
%   --and we have a monthly model, the function will return Ay+By*X0=.005
%   --unless timestep=1/12 is provided in which case it returns Ay+By*X0=.06
%
% Where under Q:
%   X(t+1) - X(t) = K0d + K1d*X(t) + eps(t+1),  cov(eps(t+1)) = H0d
%
% A1 = -rho0d
% B1 = -rho1d
% At = A(t-1) + K0d'*B(t-1) .5*B(t-1)'*H0d*B(t-1) - rho0d
% Bt = B(t-1) + K1d'*B(t-1) - rho1d
%
% mautirities: 1*M # of periods

M = length(maturities);
N = length(K0d);
Atemp = 0;
Btemp = zeros(N,1);
A = nan(1,M);
B = nan(N,M);

curr_mat = 1;
for i=1:maturities(M)
    Atemp = Atemp + K0d'*Btemp +.5*Btemp'*H0d*Btemp - rho0d;
    Btemp = Btemp + K1d'*Btemp - rho1d;
    
    if i==maturities(curr_mat)
        Ay(1,curr_mat) = -Atemp/maturities(curr_mat);
        By(:,curr_mat) = -Btemp/maturities(curr_mat);
        curr_mat = curr_mat + 1;
    end
end
    
if nargin==7
    Ay = Ay/timestep;
    By = By/timestep;
end
