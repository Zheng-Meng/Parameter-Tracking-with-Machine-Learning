function [ts_train, system_params] = func_generate_data_ks(t_start, t_end, param, bi_params,measurement_level, noise_level, solver)

tstep = 0.25;
param_len = length(param);

N = 256;

if bi_params == 1
    alpha = param;
    v = 1;

    system_params = [alpha; v*ones(1, param_len)];
elseif bi_params == 2
    alpha = 1;
    v = param;

    system_params = [alpha*ones(1, param_len); v];
end

% dynamical noise
% noise_level = 0;
t_all = (t_end - t_start);
tmax = t_all;

[Adata,X,~] = func_KSsim1D(system_params, N, tmax, tstep);

ts_train=Adata;

ts_train = ts_train + measurement_level * randn(size(ts_train));
ts_train = normalize(ts_train);

% figure('Position',[300 300 900 250])
% surf(1:size(ts_train, 2), 1:size(ts_train, 1), ts_train)
% caxis([-25 25]);
% colormap(jet)
% view(0,90)
% shading interp
% xlabel('t'); ylabel('x')
% colorbar
% set(gcf,'color','white')

end









