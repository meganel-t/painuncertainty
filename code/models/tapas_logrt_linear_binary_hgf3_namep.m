function pstruct= tapas_logrt_linear_binary_hgf3_namep(pvec)
pstruct = struct;

pstruct.be0 = pvec(1);
pstruct.be1 = pvec(2);
pstruct.be2 = pvec(3);
pstruct.be3 = pvec(4);
pstruct.ze  = exp(pvec(5));

return;
