function [ts_train, system_params] = func_generate_data_foodchain(t_start, t_end, param, bi_params, measurement_level, noise_level, solver)

% dxdt = sigma(y-x)
% dydt = x(rho - z) - y
% dzdt = xy - beta z
% default parameters: sigma 10, rho 26, beta 8/3

dt = 0.01;
param_len = length(param);

if bi_params == 1
    a = param;
    b = 1.7;
    c = 5.0;

    system_params = [a; b*ones(1, param_len); c*ones(1, param_len)];
elseif bi_params == 2
    a = 0.94;
    b = param;
    c = 5.0;

    system_params = [a*ones(1, param_len); b; c*ones(1, param_len)];
elseif bi_params == 3
    a = 0.94;
    b = 1.7;
    c = param;

    system_params = [a*ones(1, param_len); b*ones(1, param_len); c];
end

% system_params = param;

% dynamical noise
% noise_level = 0;
t_all = t_end - t_start;

if strcmp(solver, 'rk4') == 1
    x0 = [ 0.4 * rand + 0.6 ; 0.4 * rand + 0.15 ; 0.5 * rand + 0.3];
    [~,ts_train] = ode4(@(t,x, system_params) func_foodchain(t,x,system_params), 0:dt:t_all, x0, system_params);
elseif strcmp(solver, 'he2') == 1
    x0 = [ 0.4 * rand + 0.6 ; 0.4 * rand + 0.15 ; 0.5 * rand + 0.3];
    [~,ts_train] = ode2_Heun_2(@(t,x, system_params) func_foodchain(t,x,system_params), 0:dt:t_all, x0, system_params, noise_level);
end
ts_train = ts_train + measurement_level * randn(size(ts_train));

ts_train = normalize(ts_train);

end

