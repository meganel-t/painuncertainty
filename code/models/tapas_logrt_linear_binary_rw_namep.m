function pstruct= tapas_logrt_linear_binary_rw_namep(pvec)
pstruct = struct;

pstruct.be0 = pvec(1);
pstruct.be1 = pvec(2);
pstruct.ze  = exp(pvec(3));

return;
