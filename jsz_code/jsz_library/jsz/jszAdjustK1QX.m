function [K1Q_X, isTypicalDiagonal, m1] = jszAdjustK1QX(K1Q_X, eps1)
% function [K1Q_X, isTypicalDiagonal, m1] = jszAdjustK1QX(K1Q_X, eps1);
%
% This function adjusts diagonal K1Q_X to give a non-diagonal but more
% computationally tractable K1Q_X.
%
%
% K1Q_X can fall into a few cases:
%   0. diagonal
%   1. not diagonal
%   2. zero eigenvalue
%   3. near repeated roots
% In cases 1-3, the diagonal closed form solver doesn't work, so compute differently.
% In case 1-2, we will use the recursive solver, though there are more efficient methods.
% In case 3, we will add a positive number above the diagonal.  this gives a different member of the set of observationally equivalent models.
%   So for example:
%      [lambda1, 0; 0, lambda2] is replaced by [lambda1, f(lambda2-lambda1); 0, lambda2] when abs(lambda1 - lambda2)<eps0
%   By making f not just 0/1, it will help by making the likelihood
%   continuous if we parameterize by kinf. (not an issue in some cases.)
%
% We also order the diagonal of diagonal K1Q.
%
%

if nargin==1,
    eps1 = 1e-3;
end

diag_K1Q_X = diag(K1Q_X);
isDiagonal = all(all((K1Q_X==diag(diag_K1Q_X))));

% For diagonal matrix, sort the diagonal and check to see if we have near repeated roots.
if isDiagonal
    diag_K1Q_X = -sort(-diag_K1Q_X);
    K1Q_X = diag(diag_K1Q_X);
    
    hasNearUnitRoot = ~all(abs(diag_K1Q_X)>eps1); % Only applicable for diagonal
    hasNearRepeatedRoot = ~all(abs(diff(diag_K1Q_X))>eps1); % Only applicable for diagonal
    isTypicalDiagonal = isDiagonal & ~hasNearRepeatedRoot & ~hasNearUnitRoot;
else
    isTypicalDiagonal = false;
end


% If we have a near repeated root, add a constnat above the diagonal. This
% representative of the equivalence class gives easier inversion for latent
% states vs. yields.  By varying the constant 

if isDiagonal && ~isTypicalDiagonal 
    diff_diag = abs(diff(diag_K1Q_X));
    super_diag = cutoff_fun(diff_diag);
    K1Q_X(1:end-1,2:end) = K1Q_X(1:end-1,2:end) + diag(super_diag);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N = size(K1Q_X,1);
super_diagonal = K1Q_X(N+1:N+1:end);
m1 = max(find(cumprod([1,super_diagonal])));


% Cutoff function sets the super diagonal entry to something between 0 and
% 1, depending on how close the eigenvalues are.
function xc = cutoff_fun(x, eps1)
eps1 = 1e-3;
eps0 = 1e-5;
xc = 1.*(x<eps0) + ...
    (1 - (x - eps0)/(eps1 - eps0)).*(x>=eps0 & x<eps1) + ...
    0.*(x>eps1);

xc = 1.*(log(x)<log(eps0)) + ...
    (1 - (log(x) - log(eps0))/(log(eps1) - log(eps0))).*(log(x)>=log(eps0) & log(x)<log(eps1)) + ...
    0.*(log(x)>log(eps1));
xc(x==0) = 1;
