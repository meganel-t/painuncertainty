function [logp, yhat, res] = tapas_logrt_linear_binary_rw(r, infStates, ptrans)
be0  = ptrans(1);
be1  = ptrans(2);
ze   = exp(ptrans(3));

n = size(infStates,1);
logp = NaN(n,1);
yhat = NaN(n,1);
res  = NaN(n,1);

y = r.y(:,1);
y(r.irr) = [];

u = r.u(:,1);
u(r.irr) = [];


vhat = infStates(:,1);
vhat(r.irr) = [];
absda = abs(u-vhat);;


logrt = be0+be1.*absda;

reg = ~ismember(1:n,r.irr);
logp(reg) = -1/2.*log(8*atan(1).*ze) -(y-logrt).^2./(2.*ze);
yhat(reg) = logrt;
res(reg) = y-logrt;

return;
