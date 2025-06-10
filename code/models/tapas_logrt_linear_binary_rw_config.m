function c = tapas_logrt_linear_binary_rw_config
c = struct;
c.model = 'tapas_logrt_linear_binary_rw';

c.be0mu = log(760);
c.be0sa = 4;

c.be1mu = 0;
c.be1sa = 4;

c.logzemu = log(log(20));
c.logzesa = log(2);

c.priormus = [
c.be0mu,
c.be1mu,
c.logzemu
];

c.priorsas = [
c.be0sa,
c.be1sa,
c.logzesa 
];

c.obs_fun = @tapas_logrt_linear_binary_rw;

c.transp_obs_fun = @tapas_logrt_linear_binary_rw_transp;

return;
