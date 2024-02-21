% clear all
% close all
% clc
addpath('./funcs')

% choose system
system = 'foodchain';
% system = 'rossler';
% system = 'mg';
% system = 'l96';

% which parameter to track: 1, 2, 3
bi_params = 1; 

% s_n, s_w
noise_num = 5;
train_time_controller = 0;

% which cure to track: FM, sawtooth, and AM
test_curve = 'sinusodial';
% test_curve = 'linear';
% test_curve = 'sin_lin';

% input dimension
% input_params = [1];
% input_params = [2];
% input_params = [3];
input_params = [1, 2];
% input_params = [2, 3];
% input_params = [1, 2, 3];
% input_params = sort(randi([1, 40], 1, 5)); % special, only for lorenz96

% noise level
sigma_m = 0;
sigma_d = 0;

solver = 'rk4'; % solver: rk4 or he2
average_step = 100; % Dleta_T_s
N = 500; % network size
trials_num = round(3000 * 0.3); % training length

[validate_real_params_mean, validate_pred_params_mean, rmse] = ...
    func_params_extraction(system, bi_params, noise_num, train_time_controller, test_curve, input_params, N, trials_num, sigma_m, sigma_d, solver, average_step);

% plot figures
transient = 10;
figure();
hold on
plot(1:length(validate_real_params_mean), validate_real_params_mean)
plot(1:length(validate_real_params_mean), validate_pred_params_mean)
xlabel('t')
ylabel('signal')
legend('real', 'pred')

figure();
hold on
plot(transient:length(validate_real_params_mean), validate_real_params_mean(transient:end))
plot(transient:length(validate_real_params_mean), validate_pred_params_mean(transient:end))
xlabel('t')
ylabel('signal')
legend('real', 'pred')


















































