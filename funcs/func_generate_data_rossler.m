function [ts_train, system_params] = func_generate_data_rossler(t_start, t_end, param, bi_params,measurement_level, noise_level, solver)

dt = 0.01;
param_len = length(param);

if bi_params == 1
    a = param;
    b = 0.2;
    c = 5.7;

    system_params = [a; b*ones(1, param_len); c*ones(1, param_len)];
elseif bi_params == 2
    a = 0.2;
    b = param;
    c = 5.7;

    system_params = [a*ones(1, param_len); b; c*ones(1, param_len)];
elseif bi_params == 3
    a = 0.2;
    b = 0.2;
    c = param;

    system_params = [a*ones(1, param_len); b*ones(1, param_len); c];
end

% dynamical noise
% noise_level = 0;
t_all = t_end - t_start;

if strcmp(solver, 'rk4') == 1
    x0 = [ 28*rand-14; 30*rand-15; 20*rand];
    [~,ts_train] = ode4(@(t,x, system_params) func_rossler(t,x,system_params), 0:dt:t_all, x0, system_params);
elseif strcmp(solver, 'he2') == 1
    x0 = [ 28*rand-14; 30*rand-15; 20*rand];
    [~,ts_train] = ode2_Heun_2(@(t,x, system_params) func_rossler(t,x,system_params), 0:dt:t_all, x0, system_params, noise_level);
end

ts_train = ts_train + measurement_level * randn(size(ts_train));

ts_train = normalize(ts_train);

end

