function [uu,x,tt] = func_KSsim1D(system_params,N, tmax, tstep)

alpha = system_params(1, 1);
para_v = system_params(2, 1);

% alpha = 1;
% para_v = 1;

% Initial condition and grid setup
% N = 256;
x = 200*(1:N)'/N; % this is used in the plotting
a = -1;
b = 1;
u = cos(x/16); % initial condition
v = fft(u);
% Precompute various scalars for ETDRK4
% h = 0.25; % time step
h = tstep; % time step
k = [0:N/2-1 0 -N/2+1:-1]'/16; % wave numbers

uu = u; tt = 0;
% tmax = 150; 
nmax = round(tmax/h); nplt = floor((tmax/tmax)*h/h);

% L = alpha * k.^2 - para_v * k.^4; % Fourier multipliers
% g = -alpha * 0.5i*k;
% 
% E = exp(h*L); E2 = exp(h*L/2);
M = 16; % no. of points for complex means
r = exp(1i*pi*((1:M)-.5)/M); % roots of unity
% LR = h*L(:,ones(M,1)) + r(ones(N,1),:);
% Q = h*real(mean( (exp(LR/2)-1)./LR ,2));
% f1 = h*real(mean( (-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3 ,2));
% f2 = h*real(mean( (2+LR+exp(LR).*(-2+LR))./LR.^3 ,2));
% f3 = h*real(mean( (-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3 ,2));
% precalculate quantites
% for n = 1:nmax
% 
% end

% Main loop
aaa = 1;
for n = 1:nmax
    % disp(n)
    t = n*h;

    % % alpha = 1;
    % % para_v = 1;
    alpha = system_params(1, n);
    para_v = system_params(2, n);

    % Recompute L and related quantities since alpha has changed
    L = alpha * k.^2 - para_v * k.^4;
    g = -alpha * 0.5i*k;
    E = exp(h*L); 
    E2 = exp(h*L/2);
    LR = h*L(:,ones(M,1)) + r(ones(N,1),:);
    Q = h*real(mean( (exp(LR/2)-1)./LR ,2));
    f1 = h*real(mean( (-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3 ,2));
    f2 = h*real(mean( (2+LR+exp(LR).*(-2+LR))./LR.^3 ,2));
    f3 = h*real(mean( (-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3 ,2));

    Nv = g.*fft(real(ifft(v)).^2);
    a = E2.*v + Q.*Nv;
    Na = g.*fft(real(ifft(a)).^2);
    b = E2.*v + Q.*Na;
    Nb = g.*fft(real(ifft(b)).^2);
    c = E2.*a + Q.*(2*Nb-Nv);
    Nc = g.*fft(real(ifft(c)).^2);
    v = E.*v + Nv.*f1 + 2*(Na+Nb).*f2 + Nc.*f3;
    if mod(n,nplt)==0
        u = real(ifft(v));
        uu = [uu,u]; tt = [tt,t];
    end
end

uu = uu';

% surf(tt,x,uu'), shading interp, lighting phong, axis tight
% colormap(summer); set(gca,'zlim')
% light('color',[1 1 0],'position',[-1,2,2])
% material([0.30 0.60 0.60 40.00 1.00]);
% title('The Kuramoto Sivashinsky equation')
% xlabel('t'); ylabel('x')
% colorbar
% aaa = 1;

end




% % modified from github.com/E-Renshaw/kuramoto-sivashinsky/blob/master/KSequ.m
% 
% %Lx = 1; % *2*pi
% 
% nplot = round(tmax/r_step);
% 
% alpha = alpha_vec(1);
% para_v = para_v_vec(1);
% 
% x = Lx*2*pi*(1:N)'/N; % this is used in the plotting
% u = cos(x/Lx) + 0.5 * rand(N,1); % initial condition
% k = [0:N/2-1 0 -N/2+1:-1]'/Lx; % wave numbers
% L = alpha * k.^2 - para_v * k.^4; % Fourier multipliers
% g = -alpha * 0.5i*k;
% 
% M = 32; % no. of points for complex means
% E = exp(tstep*L); 
% E2 = exp(tstep*L/2);
% r = exp(1i*pi*((1:M)-.5)/M); % roots of unity
% LR = tstep*L(:,ones(M,1)) + r(ones(N,1),:);
% Q = tstep*real(mean( (exp(LR/2)-1)./LR ,2));
% f1 = tstep*real(mean( (-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3 ,2));
% f2 = tstep*real(mean( (2+LR+exp(LR).*(-2+LR))./LR.^3 ,2));
% f3 = tstep*real(mean( (-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3 ,2));
% 
% % Main loop
% uu = u; tt = 0;
% nmax = round(tmax/tstep); nplt = floor((tmax/nplot)/tstep);
% v = fft(u);
% for n = 1:nmax
%     t = n*tstep;
% 
%     alpha = alpha_vec(n);
%     para_v = para_v_vec(n);
% 
%     % Recompute L and related quantities since alpha has changed
%     L = alpha * k.^2 - para_v * k.^4;
%     g = -alpha * 0.5i*k;
%     E = exp(tstep*L); 
%     E2 = exp(tstep*L/2);
%     LR = tstep*L(:,ones(M,1)) + r(ones(N,1),:);
%     Q = tstep*real(mean( (exp(LR/2)-1)./LR ,2));
%     f1 = tstep*real(mean( (-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3 ,2));
%     f2 = tstep*real(mean( (2+LR+exp(LR).*(-2+LR))./LR.^3 ,2));
%     f3 = tstep*real(mean( (-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3 ,2));
% 
%     Nv = g.*fft(real(ifft(v)).^2); % N(un)
%     a = E2.*v + Q.*Nv; % an
%     Na = g.*fft(real(ifft(a)).^2); % N(an)
%     b = E2.*v + Q.*Na; % bn
%     Nb = g.*fft(real(ifft(b)).^2); % N(bn)
%     c = E2.*a + Q.*(2*Nb-Nv); % cn
%     Nc = g.*fft(real(ifft(c)).^2); % N(cn)
% 
%     v = E.*v + Nv.*f1 + 2*(Na+Nb).*f2 + Nc.*f3;
%     if mod(n,nplt)==0
%         u = real(ifft(v));
%         uu = [uu,u]; 
%         tt = [tt,t];
%     end
% end
% 
% uu = uu';
% end

