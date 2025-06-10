function y = tapas_logrt_linear_binary_ph_sim(r, infStates, p)
% Calculates the log-probability of log-reaction times y (in units of log-ms) according to the
% linear log-RT model developed with Louise Marshall and Sven Bestmann
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2014-2016 Christoph Mathys, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% Transform parameters to their native space
be0  = p(1);
be1 = p(2);
ze   = p(3);

% Initialize returned log-probabilities, predictions,
% and residuals as NaNs so that NaN is returned for all
n = size(infStates,1);


% Extract trajectories of interest from infStates
vhat = infStates(:,1);
al = infStates(:,3);



if isnan(r.c_sim.seed)
    rng('shuffle');
else
    rng(r.c_sim.seed);
end


% Calculate predicted log-reaction time
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
y = be0 +be1.*al + sqrt(ze)*randn(n, 1);

return;
