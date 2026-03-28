clc, clear all, close all, warning off;
% Nonlinear Discrete Time MPC — Levenberg-Marquardt Optimization
% System: x1[n+1] = 0.1 - x1^2 + x1*x2
%         x2[n+1] = -x1 + exp(-x2) + u
%         y[n]    = x1 + x2
%%
Ts = 1e-2;
T  = 20;
N  = round(T / Ts);

umin      = -1;
umax      =  1;
Ku        = 10;
Ky        = 15;
deltaumax = 0.2;
lamda     = 1;

nu_lm_init  = 1;
s           = 1;
lm_max_iter = 50;
plot_interval = 50;
% Tridiagonal delta-u penalty matrix (Ku+1 x Ku+1)
L = 2*eye(Ku+1);  L(end,end) = 1;
for k = 1:Ku
    L(k+1,k) = -1;  L(k,k+1) = -1;
end
%% Referance Signal
t_hist   = (0:N)' * Ts;
t_ref    = (0:N+Ky-2)' * Ts;
half     = floor(length(t_ref) / 2);
seg      = floor(half / 4);
ref_step = [0.4*ones(seg,1); 0.8*ones(seg,1); 1.2*ones(seg,1); 0.8*ones(half-3*seg,1)];
ref_sin  = 0.3*sin(0.5*pi*t_ref(1:length(t_ref)-half)) + 0.8;
yref     = [ref_step; ref_sin];
%% Initialization
c = [1; 1];
x = zeros(2, N+1);    x(:,1) = [0; 0];
u = zeros(Ku+1, N+1); u(:,1) = zeros(Ku+1, 1);
y = zeros(1, N+1);    y(1)   = c' * x(:,1);

figure('units','normalized','outerposition',[0 0 1 1],'color','w')
for i = 1:N

    nu_lm  = nu_lm_init;                  % reset damping, prevents carry-over
    U_k    = [u(2:end,i); u(end,i)];      % shifted warm start
    u_prev = u(1,i);

    [yhat, x_traj] = prediction_horizon(x(:,i), U_k, c, Ku, Ky);
    Jac  = analytic_jacobian(x_traj, U_k, c, Ku, Ky);

    e    = yref(i:i+Ky-1) - yhat;
    g    = -Jac'*e + lamda*L*U_k - [lamda*u_prev; zeros(Ku,1)];
    H_gn = Jac'*Jac + lamda*L;            % Gauss-Newton Hessian, positive definite
    f1   = e'*e + lamda*(U_k'*L*U_k - 2*u_prev*U_k(1));

    deltau  = zeros(Ku+1, 1);
    LC2     = 1;
    lm_iter = 0;
    while LC2
        lm_iter = lm_iter + 1;

        p      = -(H_gn + nu_lm*eye(Ku+1)) \ g;
        U_prop = U_k + p;

        [yhat_prop, ~] = prediction_horizon(x(:,i), U_prop, c, Ku, Ky);
        e_prop = yref(i:i+Ky-1) - yhat_prop;
        f2     = e_prop'*e_prop + lamda*(U_prop'*L*U_prop - 2*u_prev*U_prop(1));

        if f2 < f1 
            deltau = s * p;
            nu_lm  = max(0.1*nu_lm, 1e-12);  % good step: reduce damping
            LC2    = 0;

        elseif lm_iter >= lm_max_iter || nu_lm > 1e20
            nu_lm      = nu_lm_init;
            grad_step  = -s * g / (norm(g) + 1e-10);
            U_grad     = U_k + grad_step;
            [yhat_g,~] = prediction_horizon(x(:,i), U_grad, c, Ku, Ky);
            e_g        = yref(i:i+Ky-1) - yhat_g;
            f_g        = e_g'*e_g + lamda*(U_grad'*L*U_grad - 2*u_prev*U_grad(1));
            deltau     = (f_g < f1) * grad_step;  % accept gradient step only if it reduces cost
            LC2        = 0;

        else
            nu_lm = 10 * nu_lm;  % increase damping
        end
    end

    mu       = min(1, deltaumax / max([max(abs(deltau)), max(abs(diff(U_k + deltau)))]));
    deltau   = mu * deltau;
    u(:,i+1) = U_k + deltau;
    u(1,i+1) = max(umin, min(umax, u(1,i+1)));

    x(:,i+1) = f_sys(x(:,i), u(1,i+1));
    y(i+1)   = c' * x(:,i+1);

    if mod(i, plot_interval) == 0 || i == N
        clf;  tt = t_hist(1:i+1);

        subplot(221); hold on; grid on;
        plot(t_hist(1:N), yref(1:N), 'r', 'LineWidth', 2)
        plot(tt, y(1:i+1), 'k', 'LineWidth', 1.5)
        xlabel('t [s]'); ylabel('y')
        title(sprintf('LM-NMPC  Ky=%d | Ku=%d | \\lambda=%.3f ', Ky, Ku, lamda))
        legend('y_{ref}','y[n]');  axis([0 T min(yref)-.5 max(yref)+.5])

        subplot(222); hold on; grid on;
        plot(tt, yref(1:i+1)'-y(1:i+1), 'b', 'LineWidth', 1);  yline(0,'k--')
        xlabel('t [s]'); title('Tracking Error')

        subplot(223); hold on; grid on;
        plot(tt, u(1,1:i+1), 'b', 'LineWidth', 1)
        yline(umax,'r--','u_{max}');  yline(umin,'r--','u_{min}')
        xlabel('t [s]'); ylabel('u'); title('Input')

        subplot(224); hold on; grid on;
        plot(tt, x(1,1:i+1), 'k--', 'LineWidth', 1.5)
        plot(tt, x(2,1:i+1), 'r--', 'LineWidth', 1.5)
        xlabel('t [s]'); ylabel('x_i'); title('States');  legend('x_1','x_2')

        drawnow
    end
end

%% -----------------------------------------------------------------------
function xnew = f_sys(x, u)
xnew = [0.1 - x(1)^2 + x(1)*x(2);
       -x(1) + exp(-x(2)) + u];
end

function [yhat, x_traj] = prediction_horizon(x0, U, c, Ku, Ky)
% prediction horizon: U(1..Ku) free moves, U(Ku+1) held for remaining steps
x_traj      = zeros(2, Ky+1);
x_traj(:,1) = x0;
yhat        = zeros(Ky, 1);
for k = 1:Ky
    uk             = U(min(k, Ku+1));
    x_traj(:,k+1)  = f_sys(x_traj(:,k), uk);
    yhat(k)        = c' * x_traj(:,k+1);
end
end

function dfdx = Jf_x(x, ~)
dfdx = [-2*x(1)+x(2),  x(1);
        -1,            -exp(-x(2))];
end

function dfdu = Jf_u(~, ~)
dfdu = [0; 1];
end

function Jac = analytic_jacobian(x_traj, U, c, Ku, Ky)
% dy/dU  (Ky x Ku+1), lower-triangular, chain rule:
%   dy[k]/du[j] = dhdx * dfdx(x[k-1])*...*dfdx(x[j]) * dfdu(x[j-1])
% Held column (j=Ku+1) accumulates extra dfdu at each step beyond Ku.
Jac  = zeros(Ky, Ku+1);
dhdx = c';
for j = 1:Ku+1
    dxdu     = Jf_u(x_traj(:,j), U(j));
    Jac(j,j) = dhdx * dxdu;
    for k = j+1:Ky
        Fk = Jf_x(x_traj(:,k), U(min(k,Ku+1)));
        if j == Ku+1 && k > Ku+1
            dxdu = Fk*dxdu + Jf_u(x_traj(:,k), U(Ku+1));
        else
            dxdu = Fk * dxdu;
        end
        Jac(k,j) = dhdx * dxdu;
    end
end
end