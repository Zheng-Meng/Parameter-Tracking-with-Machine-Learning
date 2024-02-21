function [rmse] = func_train_main(hyperpara_set,n, system, bi_params, params_train_range, params_test_range)

eig_rho = hyperpara_set(1);
W_in_a = hyperpara_set(2);
a = hyperpara_set(3);
beta = 10^hyperpara_set(4);
k = n * hyperpara_set(5);
gaussian_window_noise = round(10 * hyperpara_set(6)); 
reservoir_tstep = round(hyperpara_set(7), 2);
bias = 1;
trials = round(3000 * 0.3);
reset = round(1000 * 5);
% can plot a phase diagram about this.

dt = 0.01;
noise_num = 10;
if strcmp(system, 'mg') == 1
    dim = 1;
    transient = 50;
elseif strcmp(system, 'l96') == 1
    dim = 40;
    transient = 50;
elseif strcmp(system, 'ks') == 1
    dim = 256;
    transient = 50;
    dt = 0.25;
else
    dim = 3;
    transient = 20;
end

time_interval = round(reservoir_tstep / dt);
average_step = 100;
step = average_step * time_interval;

trials = trials * dt / 0.01;

rho_min = params_train_range(1);
rho_max = params_train_range(2);
% rho_min = params_train_range(1) + (params_train_range(2) - params_train_range(1)) * train_time_controller;
% rho_max = params_train_range(2) - (params_train_range(2) - params_train_range(1)) * train_time_controller;

training_noise_length = round(trials + 100);

if noise_num == 0
    noise_set = (rho_max - rho_min) * rand(1, training_noise_length) + rho_min;
    noise_set = smoothdata(noise_set, 'gaussian', round(1 / dt));
    noise_set = smoothdata(noise_set, 'gaussian', gaussian_window_noise);

    noise_set = rescale(noise_set, rho_min, rho_max);
    noise_set = repelem(noise_set, step);
else
    noise_range = linspace(rho_min, rho_max, noise_num);
    noise_set = repelem(noise_range, 1 + round(training_noise_length / noise_num / (0.01 * average_step)));
    noise_set = noise_set(randperm(length(noise_set)));

    noise_set = repelem(noise_set, step * 0.01 / dt);
end

t_start = 0;
t_end = round(training_noise_length * time_interval);

bi_vector = noise_set;

sigma_m = 0;
sigma_d = 0;
solver = 'rk4';

if strcmp(system, 'lorenz') == 1
    [ts_train, system_params] = func_generate_data_lorenz(t_start, t_end, bi_vector, bi_params);
elseif strcmp(system, 'rossler') == 1
    [ts_train, system_params] = func_generate_data_rossler(t_start, t_end, bi_vector, bi_params);
elseif strcmp(system, 'foodchain') == 1
    judge = 1;
    while judge == 1
        [ts_train, system_params] = func_generate_data_foodchain(t_start, t_end, bi_vector, bi_params);
        judge_value = max(ts_train(end-5000:end, 3)) - min(ts_train(end-5000:end, 3));
        if max(ts_train(end-5000:end, 3)) - min(ts_train(end-5000:end, 3)) > 0.2
            judge = 2;
        end
    end
elseif strcmp(system, 'mg') == 1
    [ts_train, system_params] = func_generate_data_mg(t_start, t_end, bi_vector, bi_params);
elseif strcmp(system, 'ks') == 1
    [ts_train, system_params] = func_generate_data_ks(t_start, t_end, bi_vector, bi_params,sigma_m, sigma_d, solver);
elseif strcmp(system, 'l96') == 1
    [ts_train, system_params] = func_generate_data_lorenz96(t_start, t_end, bi_vector, bi_params, 0, 0, 'rk4');
end

% ts_train = ts_train(10000:end, :);
% 
% plot_length = 50000;
% xx = 1:plot_length;
% figure();
% hold on
% subplot(3, 1, 1)
% plot(xx, ts_train(1:plot_length, 1))
% subplot(3, 1, 2)
% plot(xx, ts_train(1:plot_length, 2))
% subplot(3, 1, 3)
% plot(xx, ts_train(1:plot_length, 3))

ts_train = ts_train(1:time_interval:end, :);
bi_vector = bi_vector(1:time_interval:end);

ts_train = ts_train(10000:end, :);
bi_vector = bi_vector(10000:end);

%% reservoir training
% rng('shuffle');

W_in = W_in_a*(2*rand(n,dim));
res_net=sprandsym(n,k/n);
eig_D=eigs(res_net,1); %only use the biggest one. Warning about the others is harmless
res_net=(eig_rho/(abs(eig_D))).*res_net;
res_net=full(res_net);
bias_matrix = bias * ones(n, 1);

train_length = round(length(bi_vector) - (transient) / dt);
washup_length = round(transient / dt);

train_x = ts_train';
train_x = train_x(:, 1:train_length);

r_all=zeros(n,train_length+1);%2*rand(n,1)-1;%
r_all(:,1) = 2*rand(n,1)-1;
for ti=1:train_length
    if mod(ti, reset) == 0
        r_all(:,ti) = 2*rand(n,1)-1;
    end
    r_all(:,ti+1)=(1 - a) * r_all(:,ti) + a * tanh(res_net*r_all(:,ti) + W_in * train_x(:,ti) + bias_matrix);
end
r_out=r_all(:,washup_length+2:end); % n * (train_length - 11)
r_out(2:2:end,:)=r_out(2:2:end,:).^2;
r_end = r_out(:, end);

r_train = [r_out; train_x(:, washup_length+1:end); ones(1, size(r_out, 2))];
y_train = bi_vector;
y_train = y_train(washup_length+1:train_length);

Wout = y_train * r_train' * (r_train * r_train' + beta * eye(n + dim + 1)) ^ (-1);

%% testing

amplitude = ((params_test_range(2) - params_test_range(1)) * 1/2);
rho_0 = ( params_test_range(1) + params_test_range(2) ) / 2;
lorenz_f = round( ((150 - 50) * rand(1) + 50) * time_interval );
if strcmp(system, 'ks') == 1
    lorenz_f = round( ((150 - 50) * rand(1) + 50) * 30  * reservoir_tstep / 0.25);
end
params_t_vali = amplitude * sin((t_start:reservoir_tstep:t_end) * 2 * pi / lorenz_f) + rho_0;

params_t_vali = func_step_average(params_t_vali, average_step);
params_t_vali = repelem(params_t_vali, step);

bi_vector = params_t_vali;
if strcmp(system, 'lorenz') == 1
    [ts_train_vali, system_params] = func_generate_data_lorenz(t_start, t_end, bi_vector, bi_params);
elseif strcmp(system, 'rossler') == 1
    [ts_train_vali, system_params] = func_generate_data_rossler(t_start, t_end, bi_vector, bi_params);
elseif strcmp(system, 'foodchain') == 1
    judge = 1;
    while judge == 1
        [ts_train_vali, system_params] = func_generate_data_foodchain(t_start, t_end, bi_vector, bi_params);
        if max(ts_train_vali(end-5000:end, 3)) - min(ts_train_vali(end-5000:end, 3)) > 0.2
            judge = 2;
        end
    end
elseif strcmp(system, 'mg') == 1
    [ts_train_vali, system_params] = func_generate_data_mg(t_start, t_end, bi_vector, bi_params);
elseif strcmp(system, 'ks') == 1
    [ts_train_vali, system_params] = func_generate_data_ks(t_start, t_end, bi_vector, bi_params,sigma_m, sigma_d, solver);
elseif strcmp(system, 'l96') == 1
    [ts_train_vali, system_params] = func_generate_data_lorenz96(t_start, t_end, bi_vector, bi_params, 0, 0, 'rk4');
end

ts_train_vali = ts_train_vali(1:time_interval:end, :);
params_t_vali = params_t_vali(1:time_interval:end);

ts_train_vali = ts_train_vali(10000:end, :);
params_t_vali = params_t_vali(10000:end);

vali_input = ts_train_vali';
vali_input = vali_input(:, 1 : end);

validate_length = length(params_t_vali);
validate_pred_params = zeros(1, validate_length);

%r=zeros(n,1);
r = 2*rand(n,1)-1;
u = zeros(dim,1);
u(:) = vali_input(:, 1);
for t_i = 1:validate_length-1
    %u(dim+1:end) = udata(tp_i,train_length+t_i,dim+1:end);
    r = (1-a) * r + a * tanh(res_net * r + W_in * u + bias_matrix);
    r_out = r;
    r_out(2:2:end,1) = r_out(2:2:end,1).^2; %even number -> squared
    r_out = [r_out; u; 1];
    predict_y = Wout * r_out;
    validate_pred_params(t_i) = predict_y;
    u(1:dim) = vali_input(:, t_i + 1);
end

validate_pred_params_mean = func_step_average(validate_pred_params, average_step);
validate_real_params_mean = func_step_average(params_t_vali, average_step);

% calibration

cali_pos_1 = round(0.25 * length(validate_real_params_mean));
cali_pos_2 = round(0.75 * length(validate_real_params_mean));

difference = mean(validate_pred_params_mean([cali_pos_1, cali_pos_2])-validate_real_params_mean([cali_pos_1, cali_pos_2]));
validate_pred_params_mean = validate_pred_params_mean - difference;

rmse_1 = sqrt(mean((validate_real_params_mean(10:end) - validate_pred_params_mean(10:end)).^2));


test_num = rand(1);
if test_num < 0.5
    rho_0 = params_test_range(1);
    rho_1 = params_test_range(2);
    params_t_vali = (rho_1 - rho_0) * (t_start:reservoir_tstep:t_end) / t_end + rho_0;
else
    rho_0 = params_test_range(2);
    rho_1 = params_test_range(1);
    params_t_vali = (rho_1 - rho_0) * (t_start:reservoir_tstep:t_end) / t_end + rho_0;

end


params_t_vali = func_step_average(params_t_vali, average_step);
params_t_vali = repelem(params_t_vali, step);

bi_vector = params_t_vali;
if strcmp(system, 'lorenz') == 1
    [ts_train_vali, system_params] = func_generate_data_lorenz(t_start, t_end, bi_vector, bi_params);
elseif strcmp(system, 'rossler') == 1
    [ts_train_vali, system_params] = func_generate_data_rossler(t_start, t_end, bi_vector, bi_params);
elseif strcmp(system, 'foodchain') == 1
    judge = 1;
    while judge == 1
        [ts_train_vali, system_params] = func_generate_data_foodchain(t_start, t_end, bi_vector, bi_params);
        if max(ts_train_vali(end-5000:end, 3)) - min(ts_train_vali(end-5000:end, 3)) > 0.2
            judge = 2;
        end
    end
elseif strcmp(system, 'mg') == 1
    [ts_train_vali, system_params] = func_generate_data_mg(t_start, t_end, bi_vector, bi_params);
elseif strcmp(system, 'ks') == 1
    [ts_train_vali, system_params] = func_generate_data_ks(t_start, t_end, bi_vector, bi_params,sigma_m, sigma_d, solver);
elseif strcmp(system, 'l96') == 1
    [ts_train_vali, system_params] = func_generate_data_lorenz96(t_start, t_end, bi_vector, bi_params, 0, 0, 'rk4');
end

ts_train_vali = ts_train_vali(1:time_interval:end, :);
params_t_vali = params_t_vali(1:time_interval:end);

ts_train_vali = ts_train_vali(10000:end, :);
params_t_vali = params_t_vali(10000:end);

vali_input = ts_train_vali';
vali_input = vali_input(:, 1 : end);

validate_length = length(params_t_vali);
validate_pred_params = zeros(1, validate_length);

%r=zeros(n,1);
r = 2*rand(n,1)-1;
u = zeros(dim,1);
u(:) = vali_input(:, 1);
for t_i = 1:validate_length-1
    %u(dim+1:end) = udata(tp_i,train_length+t_i,dim+1:end);
    r = (1-a) * r + a * tanh(res_net * r + W_in * u + bias_matrix);
    r_out = r;
    r_out(2:2:end,1) = r_out(2:2:end,1).^2; %even number -> squared
    r_out = [r_out; u; 1];
    predict_y = Wout * r_out;
    validate_pred_params(t_i) = predict_y;
    u(1:dim) = vali_input(:, t_i + 1);
end

validate_pred_params_mean = func_step_average(validate_pred_params, average_step);
validate_real_params_mean = func_step_average(params_t_vali, average_step);

% calibration

cali_pos_1 = round(0.25 * length(validate_real_params_mean));
cali_pos_2 = round(0.75 * length(validate_real_params_mean));

difference = mean(validate_pred_params_mean([cali_pos_1, cali_pos_2])-validate_real_params_mean([cali_pos_1, cali_pos_2]));
validate_pred_params_mean = validate_pred_params_mean - difference;

rmse_2 = sqrt(mean((validate_real_params_mean(10:end) - validate_pred_params_mean(10:end)).^2));


rho_0 = ( (params_test_range(2) - params_test_range(1)) * rand(1) +  params_test_range(1));
rho_1 = rho_0;
params_t_vali = (rho_1 - rho_0) * (t_start:reservoir_tstep:t_end) / t_end + rho_0;


params_t_vali = func_step_average(params_t_vali, average_step);
params_t_vali = repelem(params_t_vali, step);

bi_vector = params_t_vali;
if strcmp(system, 'lorenz') == 1
    [ts_train_vali, system_params] = func_generate_data_lorenz(t_start, t_end, bi_vector, bi_params);
elseif strcmp(system, 'rossler') == 1
    [ts_train_vali, system_params] = func_generate_data_rossler(t_start, t_end, bi_vector, bi_params);
elseif strcmp(system, 'foodchain') == 1
    judge = 1;
    while judge == 1
        [ts_train_vali, system_params] = func_generate_data_foodchain(t_start, t_end, bi_vector, bi_params);
        if max(ts_train_vali(end-5000:end, 3)) - min(ts_train_vali(end-5000:end, 3)) > 0.2
            judge = 2;
        end
    end
elseif strcmp(system, 'mg') == 1
    [ts_train_vali, system_params] = func_generate_data_mg(t_start, t_end, bi_vector, bi_params);
elseif strcmp(system, 'ks') == 1
    [ts_train_vali, system_params] = func_generate_data_ks(t_start, t_end, bi_vector, bi_params,sigma_m, sigma_d, solver);
elseif strcmp(system, 'l96') == 1
    [ts_train_vali, system_params] = func_generate_data_lorenz96(t_start, t_end, bi_vector, bi_params, 0, 0, 'rk4');
end

ts_train_vali = ts_train_vali(1:time_interval:end, :);
params_t_vali = params_t_vali(1:time_interval:end);

ts_train_vali = ts_train_vali(10000:end, :);
params_t_vali = params_t_vali(10000:end);

vali_input = ts_train_vali';
vali_input = vali_input(:, 1 : end);

validate_length = length(params_t_vali);
validate_pred_params = zeros(1, validate_length);

%r=zeros(n,1);
r = 2*rand(n,1)-1;
u = zeros(dim,1);
u(:) = vali_input(:, 1);
for t_i = 1:validate_length-1
    %u(dim+1:end) = udata(tp_i,train_length+t_i,dim+1:end);
    r = (1-a) * r + a * tanh(res_net * r + W_in * u + bias_matrix);
    r_out = r;
    r_out(2:2:end,1) = r_out(2:2:end,1).^2; %even number -> squared
    r_out = [r_out; u; 1];
    predict_y = Wout * r_out;
    validate_pred_params(t_i) = predict_y;
    u(1:dim) = vali_input(:, t_i + 1);
end

validate_pred_params_mean = func_step_average(validate_pred_params, average_step);
validate_real_params_mean = func_step_average(params_t_vali, average_step);

% calibration

cali_pos_1 = round(0.25 * length(validate_real_params_mean));
cali_pos_2 = round(0.75 * length(validate_real_params_mean));

difference = mean(validate_pred_params_mean([cali_pos_1, cali_pos_2])-validate_real_params_mean([cali_pos_1, cali_pos_2]));
validate_pred_params_mean = validate_pred_params_mean - difference;

rmse_3 = sqrt(mean((validate_real_params_mean(10:end) - validate_pred_params_mean(10:end)).^2));

%%
rmse = rmse_1 + rmse_2 + rmse_3;
end

