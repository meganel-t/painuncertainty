function pstruct= tapas_logrt_linear_binary_hgf2_namep(pvec)
pstruct = struct;

pstruct.be0 = pvec(1);
pstruct.be1 = pvec(2);
pstruct.be2 = pvec(3);
pstruct.ze  = exp(pvec(4));

return;
