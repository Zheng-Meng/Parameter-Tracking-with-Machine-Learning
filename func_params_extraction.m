function [validate_real_params_mean, validate_pred_params_mean, rmse] = func_params_extraction(system, bi_params, noise_num, train_time_controller, test_curve, input_params, N, trials_num, sigma_m, sigma_d, solver, average_step_num)

warning('off','all')
% read hyperparameters
if strcmp(system, 'rossler') == 1
    transient = 20;
    dim = 3;
    if bi_params == 1
        load('./save_opt/opt_rc_noisenum_rossler_params1_20221021T091202_366.mat')
    elseif bi_params == 2
        load('./save_opt/opt_rc_noisenum_rossler_params2_20221021T091442_843.mat')
    elseif bi_params == 3
        load('./save_opt/opt_rc_noisenum_rossler_params3_20221021T091716_562.mat')
    end
    
elseif strcmp(system, 'foodchain') == 1
    transient = 20;
    dim = 3;
    if bi_params == 1
        load('./save_opt/opt_rc_noisenum_foodchain_params1_20221110T145804_231.mat')
    elseif bi_params == 2
        load('./save_opt/opt_rc_noisenum_foodchain_params2_20221110T150639_895.mat')
    elseif bi_params == 3
        load('./save_opt/opt_rc_noisenum_foodchain_params3_20221112T102528_782.mat')
    end
    
elseif strcmp(system, 'mg') == 1
    transient = 50;
    dim = 1;
    load('./save_opt/opt_rc_noisenum_mg_params1_20221108T172302_488.mat')

elseif strcmp(system, 'ks') == 1
    transient = 50;
    dim = 256;
    load('./save_opt/opt_rc_noisenum_ks_params1_20240115T112624_899.mat')
    system = 'ks';

elseif strcmp(system, 'l96') == 1
    transient = 50;
    load('./save_opt/opt_rc_noisenum_l96_params1_20240110T163605_949.mat')
    dim = 40;
    system = 'l96';
end

hyperpara_set = opt_result;

n = N;

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
trials = round(trials_num);

dt = 0.01;
if strcmp(system, 'ks') == 1
    dt = 0.25;
    reservoir_tstep = 0.25;
end

time_interval = round(reservoir_tstep / dt);
average_step = average_step_num;
step = average_step * time_interval;

rho_min = params_train_range(1) + (params_train_range(2) - params_train_range(1)) * train_time_controller;
rho_max = params_train_range(2) - (params_train_range(2) - params_train_range(1)) * train_time_controller;

trials = trials * dt / 0.01;

training_noise_length = round(trials);

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
% bi_vector: the training parameter vector
bi_vector = noise_set;

% tic;

if strcmp(system, 'rossler') == 1
    [ts_train, system_params] = func_generate_data_rossler(t_start, t_end, bi_vector, bi_params,sigma_m, sigma_d, solver);
elseif strcmp(system, 'foodchain') == 1
    judge = 1;
    count = 1;
    while judge == 1
        [ts_train, system_params] = func_generate_data_foodchain(t_start, t_end, bi_vector, bi_params,sigma_m, sigma_d, solver);
        judge_value = max(ts_train(end-5000:end, 3)) - min(ts_train(end-5000:end, 3));
        if max(ts_train(end-5000:end, 3)) - min(ts_train(end-5000:end, 3)) > 0.2
            judge = 2;
        end
        count = count+1;
        if count > 1000
            break
        end
        
    end
elseif strcmp(system, 'mg') == 1
    [ts_train, system_params] = func_generate_data_mg(t_start, t_end, bi_vector, bi_params,sigma_m, sigma_d, solver);
elseif strcmp(system, 'ks') == 1
    [ts_train, system_params] = func_generate_data_ks(t_start, t_end, bi_vector, bi_params,sigma_m, sigma_d, solver);
elseif strcmp(system, 'l96') == 1
    [ts_train, system_params] = func_generate_data_lorenz96(t_start, t_end, bi_vector, bi_params,sigma_m, sigma_d, solver);
end

ts_train = ts_train(1:time_interval:end, :);
bi_vector = bi_vector(1:time_interval:end);

% toc;

% plot figures
% figure();
% hold on
% subplot(3,1,1);
% plot(ts_train(1:1000, 1))
% subplot(3,1,2);
% plot(ts_train(1:1000, 2))
% subplot(3,1,3);
% plot(ts_train(1:1000, 3))

% if strcmp(system, 'ks') == 1 || strcmp(system, 'l96') == 1
%     figure('Position',[300 300 900 250])
%     surf(1:size(ts_train, 1), 1:size(ts_train, 2), ts_train')
%     caxis([-25 25]);
%     colormap(jet)
%     view(0,90)
%     shading interp
%     xlabel('t'); ylabel('x')
%     colorbar
%     set(gcf,'color','white')
% end

% training data, abandon the beginning transient
ts_train = ts_train(10000:end, :);
bi_vector = bi_vector(10000:end);
bi_vector = bi_vector(1:size(ts_train, 1));

save_data.ts_train_full = ts_train;

dim = length(input_params);
ts_train = ts_train(:, input_params);

save_data.ts_train = ts_train;
save_data.bi_vector = bi_vector;
save_data.t_step = reservoir_tstep;

save_data.bi_params = bi_params;
save_data.noise_num = noise_num;
save_data.train_time_controller = train_time_controller;
save_data.test_curve = test_curve;
save_data.input_params = input_params;

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

if strcmp(test_curve, 'sinusodial') == 1
    amplitude = ((params_test_range(2) - params_test_range(1)) * 1/2);
    rho_0 = ( params_test_range(1) + params_test_range(2) ) / 2;

    lorenz_f = (normalize((t_start:reservoir_tstep:t_end), 'range', [30*time_interval, 140*time_interval]));
    params_t_vali = amplitude * sin((t_start:reservoir_tstep:t_end) * 2 * pi ./ lorenz_f) + rho_0;
    if strcmp(system, 'ks') == 1
        lorenz_f = (normalize((t_start:reservoir_tstep:t_end), 'range', [30 * 30 * reservoir_tstep / 0.25, 140 * 30 * reservoir_tstep / 0.25]));
        params_t_vali = amplitude * sin((t_start:reservoir_tstep:t_end) * 2 * pi ./ lorenz_f) + rho_0;
    end

elseif strcmp(test_curve, 'linear') == 1
    rho_0 = params_test_range(1);
    rho_1 = params_test_range(2);

    params_t_vali_1 = (rho_1 - rho_0) * (t_start:reservoir_tstep:round(t_end/4)) / round(t_end/4) + rho_0;
    params_t_vali = [flip(params_t_vali_1), params_t_vali_1, flip(params_t_vali_1), params_t_vali_1];
    
elseif strcmp(test_curve, 'sin_lin') == 1
    rho_0 = params_test_range(1);
    rho_1 = params_test_range(2);

    amplitude = ((t_start:reservoir_tstep:t_end) / t_end).^1.5;
    amplitude = normalize(amplitude, 'range');

    lorenz_f = round( ((100 - 60) * rand(1) + 60) * time_interval );
    params_t_vali = amplitude .* sin((t_start:reservoir_tstep:t_end) * 2 * pi / lorenz_f);
    params_t_vali = normalize(params_t_vali, 'range', [rho_0, rho_1]);

    if strcmp(system, 'ks') == 1
        lorenz_f = round( ((100 - 60) * rand(1) + 60) * 30 * reservoir_tstep / 0.25 );
        params_t_vali = amplitude .* sin((t_start:reservoir_tstep:t_end) * 2 * pi / lorenz_f);
        params_t_vali = normalize(params_t_vali, 'range', [rho_0, rho_1]);
    end
    
elseif strcmp(test_curve, 'constant') == 1
    rho_0 = ( (params_test_range(2) - params_test_range(1)) * rand(1) +  params_test_range(1));
    rho_1 = rho_0;
    params_t_vali = (rho_1 - rho_0) * (t_start:reservoir_tstep:t_end) / t_end + rho_0;
end

% params_t_vali = func_step_average(params_t_vali, average_step);
params_t_vali = func_step_average(params_t_vali, average_step-1);
params_t_vali = repelem(params_t_vali, step);

bi_vector = params_t_vali;
if strcmp(system, 'rossler') == 1
    [ts_train_vali, system_params] = func_generate_data_rossler(t_start, t_end, bi_vector, bi_params,sigma_m, sigma_d, solver);
elseif strcmp(system, 'foodchain') == 1
    judge = 1;
    while judge == 1
        [ts_train_vali, system_params] = func_generate_data_foodchain(t_start, t_end, bi_vector, bi_params,sigma_m, sigma_d, solver);
        if max(ts_train_vali(end-5000:end, 3)) - min(ts_train_vali(end-5000:end, 3)) > 0.2
            judge = 2;
        end
    end
elseif strcmp(system, 'mg') == 1
    [ts_train_vali, system_params] = func_generate_data_mg(t_start, t_end, bi_vector, bi_params,sigma_m, sigma_d, solver);

elseif strcmp(system, 'ks') == 1
    [ts_train_vali, system_params] = func_generate_data_ks(t_start, t_end, bi_vector, bi_params,sigma_m, sigma_d, solver);

elseif strcmp(system, 'l96') == 1
    [ts_train_vali, system_params] = func_generate_data_lorenz96(t_start, t_end, bi_vector, bi_params,sigma_m, sigma_d, solver);
end

ts_train_vali = ts_train_vali(1:time_interval:end, :);
params_t_vali = params_t_vali(1:time_interval:end);

ts_train_vali = ts_train_vali(10000:end, :);
params_t_vali = params_t_vali(10000:end);

ts_train_vali_length = size(ts_train_vali, 1);
params_t_vali_length = length(params_t_vali);
smaller_length = min(ts_train_vali_length, params_t_vali_length);

ts_train_vali = ts_train_vali(1:smaller_length, :);
params_t_vali = params_t_vali(1:smaller_length);

% ts_train_vali = ts_train_vali + sigma_m * randn(size(ts_train_vali));

ts_train_vali = ts_train_vali(:, input_params);

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

save_data.pred_params_ncali = validate_pred_params_mean;

% calibration

cali_pos_1 = round(0.25 * length(validate_real_params_mean));
cali_pos_2 = round(0.75 * length(validate_real_params_mean));

difference = mean(validate_pred_params_mean([cali_pos_1, cali_pos_2])-validate_real_params_mean([cali_pos_1, cali_pos_2]));
validate_pred_params_mean = validate_pred_params_mean - difference;

rmse = sqrt(mean((validate_real_params_mean(10:end) - validate_pred_params_mean(10:end)).^2));

save_data.real_params = validate_real_params_mean;
save_data.pred_params = validate_pred_params_mean;

% save_file
% save(['./save_for_paper/', system '_bi_' num2str(bi_params), '_' test_curve, '_cali' '.mat'], 'save_data')
% save(['./save_for_paper/', system '_bi_' num2str(bi_params), '_' test_curve, '_xy' '.mat'], 'save_data')
% 
% save(['./save_for_paper/',  'params_example_1.mat'], 'save_data')
% save(['./save_for_paper/', system '_bi_' num2str(bi_params), '_' test_curve, '_cali_dim5' '.mat'], 'save_data')
aaa = 1;

end