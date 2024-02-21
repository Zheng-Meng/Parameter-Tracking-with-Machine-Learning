function [t,x] = ode4_delay(f,t,x0,params)
%RK-4 solver for constant step length

% tau/h must be an integer!!!
tau_init = max(max(params));

h = t(2) - t(1); % step length
x = zeros(length(x0),length(t));
x(:,1:round(tau_init/h)) = repmat(x0, [1, round(tau_init/h)]); % initial period
% noise = noise_level * rand(size(x));

% index of t is the 'natural way' as the input
% index of x is moved tau/h steps
for step_i = round(tau_init/h)+1: length(t)-1
    tau = params(step_i);
    tau_integer = round(tau/h);
    
    k1 = f( t(step_i-tau_integer), x(:,step_i), x(:,step_i - tau_integer));
    k2 = f( t(step_i-tau_integer) + h/2, x(:,step_i) + h/2 * k1 , x(:,step_i - tau_integer));
    k3 = f( t(step_i-tau_integer) + h/2, x(:,step_i) + h/2 * k2 , x(:,step_i - tau_integer));
    k4 = f( t(step_i-tau_integer) + h, x(:,step_i) + h * k3 , x(:,step_i - tau_integer));
    x(:,step_i + 1) = x(:,step_i) + h/6 * (k1 + 2*k2 + 2*k3 + k4);
end

% x = x(:,round(tau_init/h)+1:end);
x = x';

end

