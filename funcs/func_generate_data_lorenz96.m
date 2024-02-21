function [ts_train, system_params] = func_generate_data_lorenz96(t_start, t_end, param, bi_params,measurement_level, noise_level, solver)

dt = 0.01;
param_len = length(param);

system_params = param;

% dynamical noise
% noise_level = 0;
t_all = t_end - t_start;

if strcmp(solver, 'rk4') == 1
    x0 = [ rand(40, 1)];
    [~,ts_train] = ode4(@(t,x, system_params) func_lorenz96(t,x,system_params), 0:dt:t_all, x0, system_params);
elseif strcmp(solver, 'he2') == 1
    x0 = [ rand(40, 1)];
    [~,ts_train] = ode2_Heun_2(@(t,x, system_params) func_lorenz96(t,x,system_params), 0:dt:t_all, x0, system_params, noise_level);
end

ts_train = ts_train + measurement_level * randn(size(ts_train));

ts_train = normalize(ts_train);

end