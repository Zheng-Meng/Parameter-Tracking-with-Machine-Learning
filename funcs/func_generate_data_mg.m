function [ts_train, system_params] = func_generate_data_mg(t_start, t_end, param, bi_params, measurement_level, noise_level, solver)

% dxdt = sigma(y-x)
% dydt = x(rho - z) - y
% dzdt = xy - beta z
% default parameters: sigma 10, rho 26, beta 8/3

dt = 0.01;
param_len = length(param);

tau = param;
mackey_beta=0.2;
mackey_gamma=0.1;
mackey_power=10;
mg_params=[mackey_beta,mackey_gamma,mackey_power];

system_params = tau;
% dynamical noise
% noise_level = 0;
t_all = t_end - t_start;

if strcmp(solver, 'rk4') == 1
    x0 = 1.0+rand(1);
    [~,ts_train] = ode4_delay(@(t,x, x_tau) func_mackey(t,x,x_tau, mg_params), 0:dt:t_all, x0, system_params);
elseif strcmp(solver, 'he2') == 1
    x0 = 1.0+rand(1);
    [~,ts_train] = ode2_Heun_2_delay(@(t,x, x_tau) func_mackey(t,x,x_tau, mg_params), 0:dt:t_all, x0, system_params, noise_level);
end

ts_train = ts_train + measurement_level * randn(size(ts_train));

ts_train = normalize(ts_train);

% ts_train = ts_train(1:1/h:end);

end