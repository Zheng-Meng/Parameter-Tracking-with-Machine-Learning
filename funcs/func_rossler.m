function dxdt = func_rossler(~,x,params)

% default: a=0.2, b=0.2, c=5.7 

a = params(1);
b = params(2);
c = params(3);


dxdt = zeros(3,1);
dxdt(1) = - (x(2) + x(3));
dxdt(2) = x(1) + a * x(2);
dxdt(3) = b + x(3) * (x(1) - c);


end


































