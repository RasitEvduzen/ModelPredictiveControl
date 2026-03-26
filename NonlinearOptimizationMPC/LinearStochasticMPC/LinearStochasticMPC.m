clc, clear all, close all, warning off;
%% Linear Discrete Time Stochastic MPC with KF State Estimator

A = [1.1 0.6; 0.2 0.1];  % System matrix
b = [1; 0];               % Input matrix
c = [1 1]';               % Output matrix

%% Algorithm Parameters
Ts = 1e-1;
T  = 40;
N  = round(T / Ts);

umin      = -1;
umax      =  1;
Ku        = 5;
Ky        = 10;
deltaumax = 0.2;
lamda     = 0.5;

plot_interval = 10;

%% Stochastic Noise Parameters
q_sys = 1e-4;   % Process noise variance    — w ~ N(0, q_sys*I)
r_sys = 5e-3;   % Measurement noise variance — v ~ N(0, r_sys)

%% KF Observer Tuning
Q_obs = q_sys * eye(2);
R_obs = r_sys;
P_obs = eye(2);

%% Time and Reference Signal  first half: multi-step, second half: sinusoidal
t_hist   = (0:N)' * Ts;
t_ref    = (0:N+Ky-2)' * Ts;
half     = floor(length(t_ref) / 2);
seg      = floor(half / 4);
ref_step = [0.4*ones(seg,1); 0.8*ones(seg,1); 1.2*ones(seg,1); 0.8*ones(half-3*seg,1)];
ref_sin  = 0.3*sin(0.5*pi*t_ref(1:length(t_ref)-half)) + 0.8;
yref     = [ref_step; ref_sin];

%% Initial Conditions
x_true  = zeros(2, N+1);    x_true(:,1)  = [0; 0];
x_hat   = zeros(2, N+1);    x_hat(:,1)   = [0; 0];
u       = zeros(Ku+1, N+1); u(:,1)       = zeros(Ku+1, 1);
y_true  = zeros(1, N+1);    y_true(1)    = c' * x_true(:,1);
y_meas  = zeros(1, N+1);    y_meas(1)    = y_true(1);
y_upper = zeros(1, N+1);
y_lower = zeros(1, N+1);
sigma_y    = sqrt(c' * P_obs * c);
y_upper(1) = c'*x_hat(:,1) + 2*sigma_y;
y_lower(1) = c'*x_hat(:,1) - 2*sigma_y;

%% MPC Matrices — computed once offline
[L, M, Z] = LDTS_MPC_matrices(A, b, c, Ku, Ky);
Hes   = M'*M + lamda*L;
Hters = inv(Hes);

%% Main Loop
figure('units','normalized','outerposition',[0 0 1 1],'color','w')
for i = 1:N
    % KF Predict
    x_hat_pred = A*x_hat(:,i) + b*u(1,i);
    P_pred     = A*P_obs*A' + Q_obs;
    y_meas(i+1) = c'*x_true(:,i) + sqrt(r_sys)*randn;     % Noisy Measurement

    % KF Update
    Inn   = y_meas(i+1) - c'*x_hat_pred;
    S_obs = c'*P_pred*c + R_obs;
    K_obs = P_pred*c / S_obs;
    x_hat(:,i+1) = x_hat_pred + K_obs*Inn;
    P_obs = (eye(2) - K_obs*c') * P_pred;

    % Confidence interval  95%: mean +/- 2sigma
    sigma_y      = sqrt(c' * P_obs * c);
    y_upper(i+1) = c'*x_hat(:,i+1) + 2*sigma_y;
    y_lower(i+1) = c'*x_hat(:,i+1) - 2*sigma_y;

    % MPC 
    g      = Hes*u(:,i) - (M'*(yref(i:i+Ky-1) - Z*x_hat(:,i+1)) + [lamda*u(1,i); zeros(Ku,1)]);
    deltau = -Hters * g;
    mu     = min(1, deltaumax / max([max(abs(deltau)), max(abs(diff(u(:,i) + deltau)))]));
    deltau = mu * deltau;
    u(:,i+1) = u(:,i) + deltau;
    u(1,i+1) = max(umin, min(umax, u(1,i+1)));

    % Real System with process noise
    x_true(:,i+1) = A*x_true(:,i) + b*u(1,i+1) + sqrt(q_sys)*randn(2,1);
    y_true(i+1)   = c'*x_true(:,i+1);

    % Plot
    if mod(i, plot_interval) == 0 || i == N
        clf

        subplot(221); hold on; grid on;
        tt = t_hist(1:i+1);
        fill([tt; flipud(tt)], ...
             [y_upper(1:i+1)'; flipud(y_lower(1:i+1)')], ...
             [0 1 0], 'EdgeColor','none', 'FaceAlpha', 0.5)
        plot(t_hist(1:N),   yref(1:N),         'r',  'LineWidth', 2.0)
        plot(tt,            y_true(1:i+1),      'k',  'LineWidth', 1.5)
        plot(tt,            c'*x_hat(:,1:i+1),  'b--','LineWidth', 1.0)
        xlabel('t [s]'); ylabel('y')
        title(sprintf('Ky=%d | Ku=%d | \\lambda=%.2f | q=%.0e | r=%.0e', Ky, Ku, lamda, q_sys, r_sys))
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
function [L, M, Z] = LDTS_MPC_matrices(A, b, c, Ku, Ky)
L = 2*eye(Ku+1);
L(end,end) = 1;
for i = 1:Ku
    L(i+1,i) = -1;
    L(i,i+1) = -1;
end
if Ku == 0, L = 1; end

Z = zeros(Ky, length(b));
for k = 1:Ky
    Z(k,:) = c'*A^k;
end

M = zeros(Ky, Ku+1);
M(1,:) = [c'*b, zeros(1,Ku)];
for i = 2:Ky
    M(i,:) = [c'*A^(i-1)*b, M(i-1,1:end-1)];
end
for k = Ku+2:Ky
    aa = 0;
    for l = 1:k-Ku-1
        aa = aa + c'*A^(l-1)*b;
    end
    M(k,Ku+1) = M(k,Ku+1) + aa;
end
end