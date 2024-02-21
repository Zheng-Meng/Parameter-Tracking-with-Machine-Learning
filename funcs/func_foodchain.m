function dxdt = func_foodchain(~,x,params)

k = params(1);
yc = params(2);
yp = params(3);

xc = 0.4;
% yc = 2.009;
xp = 0.08;
% yp = 2.876;
r0 = 0.16129;
c0 = 0.5;

dxdt = zeros(3,1);
dxdt(1) = x(1) * (1 - x(1)/k) - xc * yc * x(2) * x(1)/(x(1) + r0);
dxdt(2) = xc * x(2) * (yc * x(1)/(x(1) + r0) - 1) - xp*yp* x(3)* x(2)/(x(2) + c0);
dxdt(3) = xp * x(3) * (yp * x(2)/(x(2) + c0) - 1);


end


































