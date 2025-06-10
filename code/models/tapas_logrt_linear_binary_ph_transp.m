function [pvec, pstruct] = tapas_logrt_linear_binary_ph_transp(r, ptrans)

pvec = NaN(1,length(ptrans));
pstruct = struct;

pvec(1) = ptrans(1);
pstruct.be0 = pvec(1);

pvec(2) = ptrans(2);
pstruct.be1 = pvec(2);

pvec(3) = exp(ptrans(3));
pstruct.ze = pvec(3);
return;
