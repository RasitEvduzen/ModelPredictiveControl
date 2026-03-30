clc, clear all, close all, warning off;
% Nonlinear Discrete Time Stochastic MPC  Gauss-Newton + EKF State Estimator
% Written By: Rasit Evduzen
% Date: 29-Mar-2026
%%
Ts = 1e-2;              % sampling time [s]
T  = 20;                % total simulation time [s]
N  = round(T / Ts);     % number of simulation steps

umin      = -1;         % input lower bound        (hard saturation limit)
umax      =  1;         % input upper bound        (hard saturation limit)
Ku        = 5;          % control horizon          (number of free input moves, Ku <= Ky)
Ky        = 10;         % prediction horizon       (total steps predicted forward)
deltaumax = 0.2;        % max input rate           (|u[k] - u[k-1]| per step)
lamda     = 1;          % delta-u penalty weight   (larger = smoother input, slower tracking)

plot_interval = 100;    % live plot refresh rate [steps]

% L — tridiagonal delta-u penalty matrix (Ku+1 x Ku+1)
L = 2*eye(Ku+1);  L(end,end) = 1;
for k = 1:Ku
    L(k+1,k) = -1;  L(k,k+1) = -1;
end

%% Stochastic Noise Parameters
q_sys = 1e-4;           % Process noise variance    — w ~ N(0, q_sys*I)
r_sys = 5e-3;           % Measurement noise variance — v ~ N(0, r_sys)

%% EKF Observer Tuning
Q_obs = q_sys * eye(2);
R_obs = r_sys;
P_obs = eye(2);

%% Input Signal
t_hist   = (0:N)' * Ts;
t_ref    = (0:N+Ky-2)' * Ts;
half     = floor(length(t_ref) / 2);
seg      = floor(half / 4);
ref_step = [0.4*ones(seg,1); 0.8*ones(seg,1); 1.2*ones(seg,1); 0.8*ones(half-3*seg,1)];
ref_sin  = 0.3*sin(0.5*pi*t_ref(1:length(t_ref)-half)) + 0.8;
yref     = [ref_step; ref_sin];

%% Initial Condition
c = [1; 1];             % output matrix: y = c'*x
x_true = zeros(2, N+1);    x_true(:,1) = [0; 0];
x_hat  = zeros(2, N+1);    x_hat(:,1)  = [0; 0];
u = zeros(Ku+1, N+1);      u(:,1) = zeros(Ku+1, 1);
y_true = zeros(1, N+1);    y_true(1) = c' * x_true(:,1);
y_meas = zeros(1, N+1);    y_meas(1) = y_true(1);
y_upper = zeros(1, N+1);
y_lower = zeros(1, N+1);
sigma_y    = sqrt(c' * P_obs * c);
y_upper(1) = c'*x_hat(:,1) + 2*sigma_y;
y_lower(1) = c'*x_hat(:,1) - 2*sigma_y;

%% Nonlinear Stochastic MPC + EKF
figure('units','normalized','outerposition',[0 0 1 1],'color','w')
for i = 1:N

    % EKF Predict
    x_hat_pred = f_sys(x_hat(:,i), u(1,i));           % Nonlinear state prediction
    F_k        = Jf_x(x_hat(:,i), u(1,i));           % Jacobian at current estimate
    P_pred     = F_k * P_obs * F_k' + Q_obs;

    % Noisy Measurement
    y_meas(i+1) = c' * x_true(:,i) + sqrt(r_sys)*randn;

    % EKF Update
    Inn   = y_meas(i+1) - c' * x_hat_pred;
    S_obs = c' * P_pred * c + R_obs;
    K_obs = P_pred * c / S_obs;
    x_hat(:,i+1) = x_hat_pred + K_obs * Inn;
    P_obs = (eye(2) - K_obs*c') * P_pred * (eye(2) - K_obs*c')' + K_obs*R_obs*K_obs';


    % Confidence Interval (95%: mean +/- 2*sigma)
    sigma_y      = sqrt(c' * P_obs * c);
    y_upper(i+1) = c' * x_hat(:,i+1) + 2*sigma_y;
    y_lower(i+1) = c' * x_hat(:,i+1) - 2*sigma_y;

    % NMPC  uses estimated state x_hat
    U_k    = u(:,i);
    u_prev = U_k(1);

    [yhat, x_traj] = predict_horizon(x_hat(:,i+1), U_k, c, Ku, Ky);
    Jac = analytic_jacobian(x_traj, U_k, c, Ku, Ky);
    e   = yref(i:i+Ky-1) - yhat;

    g = -Jac'*e + lamda*L*U_k - [lamda*u_prev; zeros(Ku,1)];
    H = Jac'*Jac + lamda*L;

    mu_reg = 1e-10;
    [~, p] = chol(H);
    while p ~= 0
        H = H + mu_reg*eye(Ku+1);
        [~, p] = chol(H);
        mu_reg = mu_reg * 10;
    end

    deltau   = -H \ g;
    mu       = min(1, deltaumax / max([max(abs(deltau)), max(abs(diff(U_k + deltau)))]));
    deltau   = mu * deltau;
    u(:,i+1) = U_k + deltau;
    u(1,i+1) = max(umin, min(umax, u(1,i+1)));

    % Real System with Process Noise
    x_true(:,i+1) = f_sys(x_true(:,i), u(1,i+1)) + sqrt(q_sys)*randn(2,1);
    y_true(i+1)   = c' * x_true(:,i+1);

    %% Plot
    if mod(i, plot_interval) == 0 || i == N
        clf
        tt = t_hist(1:i+1);

        subplot(221); hold on; grid on;
        fill([tt; flipud(tt)], ...
             [y_upper(1:i+1)'; flipud(y_lower(1:i+1)')], ...
             [0 1 0], 'EdgeColor','none', 'FaceAlpha', 0.5)
        plot(t_hist(1:N), yref(1:N), 'r', 'LineWidth', 2.0)
        plot(tt, y_true(1:i+1), 'k', 'LineWidth', 1.5)
        plot(tt, c'*x_hat(:,1:i+1), 'b--', 'LineWidth', 1.0)
        xlabel('t [s]'); ylabel('y')
        title(sprintf('SNMPC+EKF  Ky=%d | Ku=%d | \\lambda=%.2f | q=%.0e | r=%.0e', Ky, Ku, lamda, q_sys, r_sys))
        legend('95% CI','y_{ref}','y_{true}','y_{hat}','Location','best')
        axis([0 T min(yref)-.5 max(yref)+.5])

        subplot(222); hold on; grid on;
        plot(tt, yref(1:i+1)' - y_true(1:i+1), 'b', 'LineWidth', 1)
        yline(0, 'k--')
        xlabel('t [s]'); title('Tracking Error')

        subplot(223); hold on; grid on;
        plot(tt, u(1,1:i+1), 'b', 'LineWidth', 1)
        yline(umax,'r--','u_{max}'); yline(umin,'r--','u_{min}')
        xlabel('t [s]'); ylabel('u'); title('Input')

        subplot(224); hold on; grid on;
        plot(tt, x_true(1,1:i+1), 'k-',  'LineWidth', 1.5)
        plot(tt, x_hat(1,1:i+1),  'b--', 'LineWidth', 1.5)
        xlabel('t [s]'); ylabel('x_1'); title('State: true vs estimated')
        legend('x_1 true','x_1 estimated')

        drawnow
    end
end

%% -----------------------------------------------------------------------
function xnew = f_sys(x, u)
xnew = [0.1 - x(1)^2 + x(1)*x(2);
       -x(1) + exp(-x(2)) + u];
end

function [yhat, x_traj] = predict_horizon(x0, U, c, Ku, Ky)
x_traj      = zeros(2, Ky+1);
x_traj(:,1) = x0;
yhat        = zeros(Ky, 1);
for k = 1:Ky
    uk            = U(min(k, Ku+1));
    x_traj(:,k+1) = f_sys(x_traj(:,k), uk);
    yhat(k)       = c' * x_traj(:,k+1);
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
Jac  = zeros(Ky, Ku+1);
dhdx = c';
for j = 1:Ku+1
    dxdu     = Jf_u(x_traj(:,j), U(j));
    Jac(j,j) = dhdx * dxdu;
    for k = j+1:Ky
        Fk = Jf_x(x_traj(:,k), U(min(k,Ku+1)));
        if j == Ku+1 && k > Ku+1
            dxdu = Fk * dxdu + Jf_u(x_traj(:,k), U(Ku+1));
        else
            dxdu = Fk * dxdu;
        end
        Jac(k,j) = dhdx * dxdu;
    end
end
end
