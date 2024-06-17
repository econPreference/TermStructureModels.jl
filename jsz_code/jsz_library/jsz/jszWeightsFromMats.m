function W = jszWeightsFromMats(mats_woe, mats)
% function W = jszWeightsFromMats(mats_woe, mats);
%
% This is an easy way to generate the weighting matrix when we measure
% specific yields without error.
%
% For example:
% mats_woe = [1,5];
% mats = [1:5}
% gives:
%       [1,0,0,0,0;
%        0,0,0,0,1]
% 

inds = ismember(mats, mats_woe);
if ~(sum(inds)==length(mats_woe))
    error('Some of the maturities without error arent in the original maturities');
end
J = length(mats);
W = eye(J);
W = W(inds,:);