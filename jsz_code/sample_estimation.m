clear all
close all
restoredefaultpath
addpath(genpath('jsz_library'))
clc

% This gives a sample estimation of an N-factor "without error" GDTSM.
% See "A New Perspective on Gaussian Dynamic Term Structure Models" by Joslin, Singleton and Zhu

% Load some data: mats (1*J) and yields (T*J)
load('sample_zeros.mat') 


% Setup format of model/data:
N = 2;
W = pcacov(cov(yields));
W = W(:,1:N)';  % N*J
cP = yields*W'; % T*N
dt = 1/12;

% Estimate the model by ML. 
help sample_estimation_fun
VERBOSE = true;
[llks, AcP, BcP, AX, BX, kinfQ, K0P_cP, K1P_cP, sigma_e, K0Q_cP, K1Q_cP, rho0_cP, rho1_cP, cP, llkP, llkQ,  K0Q_X, K1Q_X, rho0_X, rho1_X] = ...
        sample_estimation_fun(W, yields, mats, dt, VERBOSE);

