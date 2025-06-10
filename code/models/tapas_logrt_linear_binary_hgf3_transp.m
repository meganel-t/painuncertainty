function [pvec, pstruct] = tapas_logrt_linear_binary_hgf3_transp(r, ptrans)

pvec = NaN(1,length(ptrans));
pstruct = struct;

pvec(1) = ptrans(1);
pstruct.be0 = pvec(1);

pvec(2) = ptrans(2);
pstruct.be1 = pvec(2);

pvec(3) = ptrans(3);
pstruct.be2 = pvec(3);

pvec(4) = ptrans(4);
pstruct.be3 = pvec(4);

pvec(5) = exp(ptrans(5));
pstruct.ze = pvec(5);
return;
