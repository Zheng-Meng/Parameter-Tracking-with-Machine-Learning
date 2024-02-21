clear all
close all
clc
% Experiments on the success rate
addpath('funcs/')

% delete(gcp('nocreate'))
% parpool('local', 10)
iteration = 50;

system = 'rossler';
test_curve_set = {'sinusodial', 'linear', 'sin_lin'};
noise_num = 10;
partial_set = {[1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]};
partial_set = flip(partial_set);
train_time_controller = 0;

sigma_m = 0;
sigma_d = 0;
solver = 'rk4';
average_step_num = 100;
N = 500;
trials_num = round(3000 * 0.3);


for bi_params = 1:3
    for test_curve_idx = 1:length(test_curve_set)
        test_curve = char(test_curve_set(test_curve_idx));
        rmse_matrix = zeros(length(partial_set), iteration);
        
        for partial_idx = 1:length(partial_set)
            partial = partial_set(partial_idx);
            input_params = cell2mat(partial);
            rmse_set = zeros(1, iteration);
            for iter = 1:iteration
                rmse = nan;
                while(isnan(rmse))
                    rng(iter*20000 + (now*1000-floor(now*1000))*100000)
                    [~, ~, rmse] = func_params_extraction_test(system, bi_params, noise_num, train_time_controller, test_curve, input_params, N, trials_num, sigma_m, sigma_d, solver, average_step_num);
                end
                rmse_set(iter) = rmse;
            end
            rmse_set = sort(rmse_set);
            rmse_matrix(partial_idx, :) = rmse_set;
            
            aaa = 1;
        end
        
        system_record.(['bi_params_' num2str(bi_params) '_curve_' char(test_curve)]) = rmse_matrix;
        aaa = 1;
    end
end

save(['./save_matrix/' system '_partial_iter_', datestr(now, 'mmdd'), '.mat'])






































