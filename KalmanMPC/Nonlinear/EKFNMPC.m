clc; clear all; close all; warning off;
% Nonlinear Discrete-Time MPC EKF-Based Optimization
% Written By: Rasit
% Date: 28.03.2026
% System: x1[n+1] = 0.1 - x1^2 + x1*x2
%         x2[n+1] = -x1 + exp(-x2) + u
%         y[n]    = x1 + x2
%%
c = [1; 1];

Ts = 1e-2;
T  = 20;
N  = round(T / Ts);

Ku        = 10;
Ky        = 15;
umin      = -1;
umax      =  1;
deltaumax = 0.5;

plot_interval = 100;

%% Kalman Tuning
% U is treated as the hidden state, tracking error as the innovation.
% Larger q → bigger K → aggressive update
% Larger r → smaller K → smooth update
q  = 1e-2;
r  = 1e-2;
p0 = 1.0;
I_k = eye(Ku+1);
P   = p0 * I_k;
Q   = q  * I_k;

%% Reference Signal
t_hist   = (0:N)' * Ts;
t_ref    = (0:N+Ky-2)' * Ts;
half     = floor(length(t_ref) / 2);
seg      = floor(half / 4);
ref_step = [0.4*ones(seg,1); 0.8*ones(seg,1); 1.2*ones(seg,1); 0.8*ones(half-3*seg,1)];
ref_sin  = 0.3*sin(0.5*pi*t_ref(1:length(t_ref)-half)) + 0.8;
yref     = [ref_step; ref_sin];

%% Initialization
x      = zeros(2, 1);
U      = zeros(Ku+1, 1);
x_hist = zeros(2, N+1);   x_hist(:,1) = x;
u_hist = zeros(1, N);
y_hist = zeros(1, N+1);   y_hist(1)   = c' * x;

figure('units','normalized','outerposition',[0 0 1 1],'color','w')
%% Main Loop
for i = 1:N

    % Nonlinear prediction and analytic Jacobian at current operating point
    [yhat, x_traj] = prediction_horizon(x, U, c, Ku, Ky);
    H              = analytic_jacobian(x_traj, U, c, Ku, Ky);
    e      = yref(i:i+Ky-1) - yhat;
    P_pred = P + Q;

    % EKF update 
    S = H*P_pred*H' + r*eye(Ky);
    K = P_pred*H' / S;
    U = U + K*e;
    P = (I_k - K*H)*P_pred*(I_k - K*H)' + K*(r*eye(Ky))*K';

    % Input constraints
    U = max(min(U, umax), umin);
    if i > 1
        mu_val = min(deltaumax / max(abs(diff([u_hist(i-1); U]))), 1);
        U      = U * mu_val;
    end

    % Apply first input, propagate system
    u_apply        = U(1);
    x              = f_sys(x, u_apply);
    x_hist(:,i+1)  = x;
    u_hist(i)      = u_apply;
    y_hist(i+1)    = c' * x;

    % Shifted warm start
    U = [U(2:end); U(end)];

    if mod(i, plot_interval) == 0 || i == N
        clf;  tt = t_hist(1:i+1);

        subplot(221); hold on; grid on;
        plot(t_hist(1:N), yref(1:N), 'r', 'LineWidth', 1.5)
        plot(tt, y_hist(1:i+1), 'k', 'LineWidth', 1.5)
        xlabel('t [s]'); ylabel('y')
        title(sprintf('EKF-NMPC  Ky=%d | Ku=%d | q=%.0e | r=%.0e', Ky, Ku, q, r))
        legend('y_{ref}','y[n]');  axis([0 T min(yref)-.5 max(yref)+.5])

        subplot(222); hold on; grid on;
        plot(t_hist(1:i), yref(1:i)'-y_hist(1:i), 'b', 'LineWidth', 1);  yline(0,'k--')
        xlabel('t [s]'); title('Tracking Error')

        subplot(223); hold on; grid on;
        plot(t_hist(1:i), u_hist(1:i), 'b', 'LineWidth', 1)
        yline(umax,'r--','u_{max}');  yline(umin,'r--','u_{min}')
        xlabel('t [s]'); ylabel('u'); title('Input')

        subplot(224); hold on; grid on;
        plot(tt, x_hist(1,1:i+1), 'k--', 'LineWidth', 1.5)
        plot(tt, x_hist(2,1:i+1), 'r--', 'LineWidth', 1.5)
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
% Forward rollout: U(1..Ku) free moves, U(Ku+1) held for remaining steps
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