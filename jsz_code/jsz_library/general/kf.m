function [x_tm1t, P_tm1t, x_tt, P_tt, K_t, llks] = kf(y, Phi, alpha, A, b, Q, R, x00, P00)
% function [x_tm1t, P_tm1t, x_tt, P_tt, K_t, llks] = kf(y, Phi, alpha, A, b,  Q, R, x00, P00)
% 
% Notation is as in Time Series Analysis and Its Applications by Shumway and Stoffer
% MODIFIED TO HAVE INTERCEPTS
% 
% x_t = Phi*x_{t-1} + alpha + w_t, cov(w)=Q,     x is p-dimensional
% y_t = A*x_t + b + v_t,           cov(v)=R,     y is q-dimensional
%
% y     : q*n
% Phi   : p*1
% alpha : p*1
% A     : q*p
% b     : q*1
% Q     : p*p
% R     : q*q
% x00   : p*1  x0 is assued to be normal N(x00,P00)
% P00   : p*p
%
% x_tm1t : p*n
% P_tm1t : p*p*n
% x_tt   : p*n
% P_tt   : p*p*n
% K_t    : p*q*n
% llks   : n*1
%
% P_tm1t(tau) = P_{tau}^{tau-1}, tau=1,..,n
% x_tm1t(tau) = x_{tau}^{tau-1}, tau=1,..,n
% P_tt(tau) = P_{tau}^{tau},     tau=1,..,n
% x_tt(tau) = x_{tau}^{tau},     tau=1,..,n
% K_t(tau)  = K_{tau},           tau=1,..,n Kalman Gain
% llks(t) = log(likelihood(y(t)|y(1),...y(t-1)))  (includes 2*pi terms).
%           it is likelihood not minus log likelihood

p = size(Q,1);
[q,n] = size(y);


P_tm1t = nan(p,p,n);
x_tm1t = nan(p,n);
P_tt = nan(p,p,n);
x_tt = nan(p,n);
K_t = nan(p,q,n);
llks = nan(n,1);

% Initialize the recursion and do first step:
Ip = eye(p);
x_tm1t(:,1) = Phi*x00 + alpha; % 4.35
P_tm1t(:,:,1) = Phi*P00*Phi.' + Q; % 4.36

for t=1:n
    if t==1
        x_tm1t(:,1) = Phi*x00 + alpha; % 4.35
        P_tm1t(:,:,1) = Phi*P00*Phi.' + Q; % 4.36
    else
        x_tm1t(:,t) = Phi*x_tt(:,t-1) + alpha; % 4.35
        P_tm1t(:,:,t) = Phi*P_tt(:,:,t-1)*Phi.' + Q; % 4.36
    end

    epst = (y(:,t) - (A*x_tm1t(:,t) + b)); % 4.40 (modified)
    Sigmat = A*P_tm1t(:,:,t)*A.' + R; % 4.41; 
    
    K_t(:,:,t) = P_tm1t(:,:,t)*A.'/Sigmat; % 4.39
    x_tt(:,t) = x_tm1t(:,t) + K_t(:,:,t)*epst; % 4.37
    P_tt(:,:,t) = (Ip - K_t(:,:,t)*A)*P_tm1t(:,:,t); % 4.38
    
    % NOTE: For strange parameters, we might have non-psd Sigmat which would be a problem
    term2 = log(det(Sigmat));
    term3 = max(epst.'*(Sigmat\epst),0);
%     if ~isreal(term3) || ~isreal(term2), keyboard, end
    llks(t) = -q/2*log(2*pi) - .5*term2 -.5*term3; % as in 4.67
end

