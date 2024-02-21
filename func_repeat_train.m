function mean_rmse = func_repeat_train(hyperpara_set,N,repeat_num,take_num, system, bi_params, params_train_range, params_test_range)
tic

rmse_set = zeros(repeat_num,1);
parfor repeat_i = 1:repeat_num
    rng(repeat_i*20000 + (now*1000-floor(now*1000))*100000)
    [rmse_set(repeat_i)] = func_train_main(hyperpara_set,N, system, bi_params, params_train_range, params_test_range);
end

rmse_set = sort(rmse_set);
rmse_set = rmse_set(1:take_num);

mean_rmse = mean(rmse_set);
fprintf('\nmean rmse is %f\n',mean_rmse)
% fprintf('hp %f',hyperpara_set)

fprintf('rho %f', hyperpara_set(1))
fprintf(' sigma %f', hyperpara_set(2))
fprintf(' alpha %f', hyperpara_set(3))
fprintf(' beta %f', hyperpara_set(4))
fprintf(' k %f', hyperpara_set(5))
fprintf(' gaussian1 %f', hyperpara_set(6))
fprintf(' reservoir tstep %f', hyperpara_set(7))
fprintf('\n')

toc
end