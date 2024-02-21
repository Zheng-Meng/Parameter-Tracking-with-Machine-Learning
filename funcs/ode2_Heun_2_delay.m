function [t,x] = ode2_Heun_2_delay(f,t,x0, params, sigma)
% Van den Broeck, Christian, et al. "Nonequilibrium phase transitions induced by multiplicative noise." Physical Review E 55.4 (1997): 4084
% dx/dt = f(t,x) + yita(t), without g
tau_init = max(max(params));
h = t(2) - t(1); % step length
% sigma is a column vector (for each dim)
d = length(x0);
x = zeros(d,length(t));
x(:,1:round(tau_init/h)) = repmat(x0, [1, round(tau_init/h)]); % initial period

for step_i = round(tau_init/h)+1: length(t)-1
    tau = params(step_i);
    tau_integer = round(tau/h);
    
    w = sqrt(h) * sigma * randn(d,1);
    y = x(:,step_i) + h * f(t(step_i-tau_integer), x(:,step_i), x(:,step_i - tau_integer)) + w;
    x(:,step_i + 1) = x(:,step_i) + h/2 * ( f(t(step_i-tau_integer),x(:,step_i), x(:,step_i - tau_integer)) + f(t(step_i-tau_integer),y, x(:,step_i - tau_integer)) ) + w;
end

x = x';

end

