function [t,x] = ode2_Heun_2(f,t,x0, params, sigma)
% Van den Broeck, Christian, et al. "Nonequilibrium phase transitions induced by multiplicative noise." Physical Review E 55.4 (1997): 4084
% dx/dt = f(t,x) + yita(t), without g

% sigma is a column vector (for each dim)
d = length(x0);
x = zeros(d,length(t));
x(:,1) = x0;

h = t(2) - t(1); % step length

for step_i = 1: length(t)-1
    w = sqrt(h) * sigma * randn(d,1);
    y = x(:,step_i) + h * f(t(step_i),x(:,step_i), params(:, step_i)) + w;
    x(:,step_i + 1) = x(:,step_i) + h/2 * ( f(t(step_i),x(:,step_i), params(:, step_i)) + f(t(step_i),y, params(:, step_i)) ) + w;
end

x = x';

end

