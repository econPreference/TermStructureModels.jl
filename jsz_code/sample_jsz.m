clear all
close all
restoredefaultpath
addpath(genpath('jsz_library'))
clc


load('sample_RY_model_jsz.mat')
load('sample_zeros.mat')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute the loadings and rotated model with the yields as states
[BcP, AcP, K0Q_cP, K1Q_cP, rho0_cP, rho1_cP, K0Q_X, K1Q_X, AX, BX, Sigma_X] = jszLoadings(W, K1Q_X, kinfQ, Sigma_cP, mats, dt);

yields_m = ones(length(dates),1)*AcP + (yields*W.')*BcP;

figure(1)
plot(year(dates) + month(dates)/12, yields_m)
xlabel('date')
ylabel('yields')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% We can also re-parameterize kinfQ with rho0_cP
% This is beneficial since it will help make the likelihood continuous.
[BcP, AcP, K0Q_cP, K1Q_cP, rho0_cP, rho1_cP, K0Q_X, K1Q_X, AX, BX, Sigma_X] = jszLoadings_rho0cP(W, K1Q_X, rho0_cP, Sigma_cP, mats, dt);

% We could also use rinfQ (the long run mean of the short rate under Q) instead of kinfQ provided that the model is Q-stationary.
% This will make the likelihood continuous and provides a proper bijective identification for the set of Q-stationary models
[BcP, AcP, K0Q_cP, K1Q_cP, rho0_cP, rho1_cP, K0Q_X, K1Q_X, AX, BX, Sigma_X] = jszLoadings(W, K1Q_X, -K1Q_X(1,1)*rinfQ, Sigma_cP, mats, dt);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Can compute the likelihood with the "without error assumption"
llk = jszLLK(yields, W, K1Q_X, kinfQ, Sigma_cP, mats, dt);

% Or all the bells and whistles
[llk, AcP, BcP, AX, BX, K0P_cP, K1P_cP, sigma_e, K0Q_cP, K1Q_cP, rho0_cP, rho1_cP, cP, llkP, llkQ] = ...
    jszLLK(yields, W, K1Q_X, kinfQ, Sigma_cP, mats, dt);
fprintf('The average (negative) log likelihood is %6.6g\n', mean(llk))

% With the without error assumption, we can concentrate out kinf (which
% makes the choice of rinf vs rho_cP vs kinf irrelevant)
[llk, AcP, BcP, AX, BX, kinf, K0P_cP, K1P_cP, sigma_e, K0Q_cP, K1Q_cP, rho0_cP, rho1_cP, cP, llkP, llkQ,  K0Q_X, K1Q_X, rho0_X, rho1_X] = ...
    jszLLK_kinf_conc(yields, W, K1Q_X, Sigma_cP, mats, dt);
fprintf('The average (negative) log likelihood is (concentrate) %6.6g\n', mean(llk))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The model was estimated assuming the 6-month, 2-year, 10-year zeros were
% measured without error.  We can compute the likelihood with the KF error
% assumption.  (this function requires [K0P_cP, K1P_cP, sigma_e])
[llk, AcP, BcP, AX, BX, K0Q_cP, K1Q_cP, rho0_cP, rho1_cP, cP, yields_filtered, cP_filtered] = ...
    jszLLK_KF(yields, W, K1Q_X, kinfQ, Sigma_cP, mats, dt, K0P_cP, K1P_cP, sigma_e);
fprintf('The average (negative) log likelihood is %6.6g when using KF instead of assuming yields without error at these estimates\n', mean(llk))

figure(2)
plot(year(dates) + month(dates)/12, [yields*W', cP_filtered])
xlabel('date')
ylabel('yields')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




