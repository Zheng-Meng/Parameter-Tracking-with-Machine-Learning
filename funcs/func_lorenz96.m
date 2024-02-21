function dxdt = func_lorenz96(t,x,params)

N = length(x);
F = params;
dxdt = zeros(N, 1);

for i = 1:N
    if i == 1
        dxdt(i) = (x(i+1)-x(N-1)) * x(N) - x(i) + F;
    elseif i == 2
        dxdt(i) = (x(i+1)-x(N)) * x(i-1) - x(i) + F;
    elseif i == N
        dxdt(i) = (x(1)-x(i-2)) * x(i-1) - x(i) + F;
    else
        dxdt(i) = (x(i+1)-x(i-2)) * x(i-1) - x(i) + F;
    end
end

end

