clear all
close all
clc

% Bayesian optimization to find optimal hyperparameters.

addpath('./funcs')
% 
% delete(gcp('nocreate'))
% parpool('local', 4)

iter_max = 300;
% n = 500;
% take_num = 10;
% repeat_num = 16;
n = 500;
take_num = 7;
repeat_num = 10;

% system = 'rossler';
% bi_params = 1;
% params_train_range = [0.1, 0.35];
% params_test_range = [0.1, 0.35];
% bi_params = 2;
% params_train_range = [0.1, 1.5];
% params_test_range = [0.1, 1.5];
% bi_params = 3;
% params_train_range = [60, 80];
% params_test_range = [60, 80];

% system = 'lorenz';
% bi_params = 1;
% params_train_range = [8, 15];
% params_test_range = [8, 15];
% bi_params = 2;
% params_train_range = [130, 180];
% params_test_range = [130, 180];
% bi_params = 3;
% params_train_range = [0.5, 3.5];
% params_test_range = [0.5, 3.5];

% system = 'foodchain';
% bi_params = 1;
% params_train_range = [0.9, 1];
% params_test_range = [0.9, 1];
% bi_params = 2;
% params_train_range = [1.6, 1.8];
% params_test_range = [1.6, 1.8];
% bi_params = 3;
% params_train_range = [6, 8];
% params_test_range = [6, 8];

% system = 'mg';
% bi_params = 1;
% params_train_range = [14, 22];
% params_test_range = [14, 22];

% system = 'l96';
% bi_params = 1;
% params_train_range = [8, 14];
% params_test_range = [8, 14];

system = 'ks';
bi_params = 1;
params_train_range = [0.1, 1.0];
params_test_range = [0.1, 1.0];

% set parameters range for the algorithm to explore
% 1~2: eig_rho W_in_a a beta k gaussian_window_noise 
% reservoir tstep(lorenz:0.16, rossler:0.14)
% lb = [0 0 0 -8 0 0.1, 0.1];
% ub = [3 5 1 -2 1 0.1, 2.0];
% for l96
% lb = [0 0 0 -8 0 0.1, 0.01];
% ub = [3 5 1 -2 1 0.1, 2.0];
% for ks
lb = [0 0 0 -8 0 0.1, 0.50];
ub = [3 5 1 -2 1 0.1, 0.50];

rng((now*1000-floor(now*1000))*100000)
tic
options = optimoptions('surrogateopt','MaxFunctionEvaluations',iter_max,'PlotFcn','surrogateoptplot');
filename = ['./save_opt/opt_rc_noisenum_' system '_params', num2str(bi_params), '_tstep0.5_' datestr(now,30) '_' num2str(randi(999)) '.mat'];

func = @(x) (func_repeat_train(x,n,repeat_num,take_num, system, bi_params, params_train_range, params_test_range));
[opt_result,opt_fval,opt_exitflag,opt_output,opt_trials] = surrogateopt(func,lb,ub,options);
toc

save(filename)










































