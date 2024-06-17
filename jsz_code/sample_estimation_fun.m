function [llks, AcP, BcP, AX, BX, kinfQ, K0P_cP, K1P_cP, sigma_e, K0Q_cP, K1Q_cP, rho0_cP, rho1_cP, cP, llkP, llkQ,  K0Q_X, K1Q_X, rho0_X, rho1_X] = ...
        sample_estimation_fun(W, yields, mats, dt, VERBOSE)
% function [llks, AcP, BcP, AX, BX, kinfQ, K0P_cP, K1P_cP, sigma_e, K0Q_cP, K1Q_cP, rho0_cP, rho1_cP, cP, llkP, llkQ,  K0Q_X, K1Q_X, rho0_X, rho1_X] = ...
%        sample_estimation(W, yields, mats, dt, VERBOSE)
%
% Estimates the model with the following setup:
%  1. Optimize over (lamQ, Sigma_cP).
%  2. lamQ is assumed to be real, parameterized by the difference to maintain order.
%  3. Sigma_cP parameterized by cholesky factorization in optimization
%  4. Take randomized lamQ as initial seeds.  See lines 107-108 for the randomization
%  5. always use OLS estimate of Sigma_cP to start (see Joslin, Singleton, Le)
%  6. Generate 50 random seeds and take best to start (avoid really bad areas -- no point in looking here)
%  7. (kinfQ, sigma_e, K0P, K1P) are all concentrated out of the likelihood function.  See JSZ and JLS.
%  8. Run fmincon and repeat 3 times.  Repeating re-sets the iteratively computed Hessian.
%
%
%
% INPUTS:
% W       : N*J,  weights for the yield portfolios measured without error
% yields  : T*J,  annualized zero coupon yields
% mats    : 1*J,  maturities, in years 
% dt      : scalar, time in years for each period
% VERBOSE : boolean, true prints more output
%
% OUTPUTS:
% K0Q_X      : N*1,      normalized latent-model matrix 
% K1Q_X      : N*N,      normalized latent-model matrix 
% Sigma_cP   : N*N,      positive definite matrix that is the covariance of innovations to cP
% K0P_cP     : N*1,      
% K1P_cP     : N*N,      
% sigma_e    : scalar,   standard error of yield observation errors (errors are i.i.d)
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
    


warning off all

if ~exist('VERBOSE','var') || isempty(VERBOSE), VERBOSE = true; end
nSeeds = 50;  % Number of random starting points.  We want to avoid really bad starting values.
mlam = .95;   % most negative eigenvalue is greater than -mlam
nRepeats = 3; % We run fmincon this many times in a row.  This is useful to reset the iterative computation of the Hessian


[N,J] = size(W);
cP = yields*W'; % T*N

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup the likelihood function that input a P*1 vector of parameters.
%   Parameterize eigenvalues in terms of the difference to maintain order.
%   We parametrize Sigma_cP in terms of cholesky factorization.
%   Also modify the likelihood function to return a default value for weird parameter values with numerical issues

llk_fun = @(dlamQ, cholSigma_cP) llk_fun0(yields, W, dlamQ, cholSigma_cP, mats, dt);

% dlamQ        : N*1
% cholSigma_cP : [N*(N+1)/2]*1 vector of subdiagonal element of cholesky factorization 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SETUP INITIAL CONDITIONS:
% STARTING POINT FOR Sigma_cP:
%    Always initialize Sigma_cP at the VAR estimate.  This should be accurate, see Joslin, Le, and Singleton.
[Gamma_hat, alpha_hat, Omega_hat] = regressVAR(cP);
Sigma_cP0 = Omega_hat; 
L0 = chol(Sigma_cP0, 'lower');
inds = find(tril(ones(N)));
cholSigma_cP0 = L0(inds);


% STARTING POINT FOR lamQ:
%    Generate some random seeds so we don't waste time searching with very weird parameters.
bestllk = inf; 
for n=1:nSeeds
    % To be sure the eigenvalues are ordered, we parameterize the difference in eigenvalues, dlamQ.
    dlamQ(1,1) = .01*randn;  % When this is positive we'll have Q-non-stationary model
    dlamQ(2:N,1) = -diff(sort([dlamQ(1); rand(N-1,1)]));
    llk = llk_fun(dlamQ, cholSigma_cP0);
    if llk<bestllk
        if VERBOSE, fprintf('Improved seed llk to %5.5g\n',llk), end
        bestllk = llk;
        dlamQ0 = dlamQ;
    end
end
X0 = [dlamQ0; cholSigma_cP0]; % [N + N*(N+1)/2]*1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Let's constrain so the most negative eigenvalue (=sum(dlamQ)) is greater than -mlam
A = [-ones(1,N), zeros(1,N*(N+1)/2)];
B = mlam;
Aeq = [];
Beq = [];

% Bounds for eigenvalues:
LB = [-2,-inf*ones(1,N-1)];
UB = [.5,zeros(1,N-1)];

% Bounds for cholesky factorization of Sigma_cP
A0 = ones(N);
inds_diag    = find(ismember(find(tril(A0)), find(diag(diag(A0))))) + N;
inds_offdiag = find(~ismember(find(tril(A0)), find(diag(diag(A0))))) + N;
LB(inds_diag) = 1e-7;  % Avoid getting non-singular Sigma_cP, should be positive to be identified
LB(inds_offdiag) = -inf;
UB(N+1:N*(N+1)/2+N) = inf;


options = optimset('display','off','TolX',1e-8,'TolFun',1e-8);

X = X0;
for i=1:nRepeats
    [X, llk] = FMINCON(@(Z) llk_fun(Z(1:N), Z(N+1:end)),X,A,B,Aeq,Beq,LB,UB,[],options);
    if VERBOSE
        fprintf('Likelihood on step %d: %10.10g\tparameters:',i,llk)
        fprintf('%3.3g\t',X)
        fprintf('\n')
    end
end
[llk, K1Q_X, Sigma_cP] = llk_fun0(yields, W, X(1:N), X(N+1:end), mats, dt);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


[llks, AcP, BcP, AX, BX, kinfQ, K0P_cP, K1P_cP, sigma_e, K0Q_cP, K1Q_cP, rho0_cP, rho1_cP, cP, llkP, llkQ,  K0Q_X, K1Q_X, rho0_X, rho1_X] = ...
     jszLLK_kinf_conc(yields, W, K1Q_X, Sigma_cP, mats, dt);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Likelihoood function.  Very extreme parameters may have numerical
% problems since some intermediate matrices may be nearly non-singular.  In
% this case set the likelihood to a "bad" default value.
function [llk, K1Q_X, Sigma_cP] = llk_fun0(yields, W, dlamQ, cholSigma_cP, mats, dt)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extract the vector parameters:
N = length(dlamQ);
K1Q_X = diag(cumsum(dlamQ));
inds = find(tril(ones(N)));
L(inds) = cholSigma_cP;
L = reshape(L, [N,N]);
Sigma_cP = L*L';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


default_llk = 1000;
try
    llk = mean(jszLLK_kinf_conc(yields, W, K1Q_X, Sigma_cP, mats, dt));
    if isnan(llk) || ~isreal(llk) || ~isfinite(llk)
        llk = default_llk;
    end
catch
	llk = default_llk;
end

if llk<-100
    [llks, AcP, BcP, AX, BX, kinfQ, K0P_cP, K1P_cP, sigma_e, K0Q_cP, K1Q_cP, rho0_cP, rho1_cP, cP, llkP, llkQ,  K0Q_X, K1Q_X, rho0_X, rho1_X] = ...
        jszLLK_kinf_conc(yields, W, K1Q_X, Sigma_cP, mats, dt);
    keyboard
end
    