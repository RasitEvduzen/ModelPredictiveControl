clc; clear all; close all; warning off;
% Written By: Rasit
% Linear Discrete-Time MPC & Kalman Filter Based Optimization
% Stochastic MPC — process + measurement noise with KF state estimator
% Date: 26.03.2026

A = [1.1 0.6; 0.2 0.1];  % System Matrix
b = [1; 0];               % Input Matrix
c = [1 1]';               % Output Matrix

Ts = 1e-1;   % sampling time [s]
T  = 40;     % total simulation time [s]
N  = round(T / Ts);

Ku        = 5;    % control horizon     (number of free input moves optimized at each step)
Ky        = 10;   % prediction horizon  (number of future output steps considered) (Ky >= Ku)
umin      = -1;   % input lower bound   (hard constraint, applied after every update)
umax      =  1;   % input upper bound   (hard constraint, applied after every update)
deltaumax = 0.2;  % max input rate      (maximum allowed change in u between consecutive steps)

plot_interval = 10;   % plot refresh rate (steps)
%% MPC Kalman Tuning  (U optimizer)
q  = 1e-3;   % process noise covariance   — larger q → bigger K → aggressive update (analogous to small lambda)
r  = 1e-2;    % measurement noise covariance — larger r → smaller K → smooth update (analogous to large lambda)
p0 = 1.0;    % initial state covariance   — larger p0 → faster adaptation in early steps

%% Stochastic Noise Parameters
q_sys = 1e-4;   % process noise intensity   — variance of w ~ N(0, q_sys*I) added to state
r_sys = 1e-4;   % measurement noise intensity — variance of v ~ N(0, r_sys) added to output

%% KF Observer Tuning  (state estimator)
p0_obs = 1.0;   % initial observer covariance

%% Reference Signal — first half: multi-step, second half: sinusoidal
t    = (0:N+Ky-2)' * Ts;
half = floor(length(t)/2);
seg  = floor(half/4);
ref_step = [0.4*ones(seg,1); 0.8*ones(seg,1); 1.2*ones(seg,1); 0.8*ones(half-3*seg,1)];
ref_sin  = 0.3*sin(0.5*pi*t(1:length(t)-half)) + 0.8;
yref     = [ref_step; ref_sin];

%% Offline: MPC Matrices
[M, Z] = mpc_matrices(A, b, c, Ku, Ky);
H   = M;
I_k = eye(Ku+1);

%% Initial Conditions
x_true = [0; 0];   % true (hidden) state
x_hat  = [0; 0];   % KF observer estimate
U      = zeros(Ku+1, 1);
P      = p0 * eye(Ku+1);        % MPC KF covariance
P_obs  = p0_obs * eye(2);       % Observer KF covariance
Q_mpc  = q*I_k;

t_hist      = (0:N) * Ts;
x_true_hist = zeros(2, N+1);   x_true_hist(:,1) = x_true;
x_hat_hist  = zeros(2, N+1);   x_hat_hist(:,1)  = x_hat;
u_hist      = zeros(1, N);
y_hist      = zeros(1, N+1);   y_hist(1) = c'*x_true;
y_meas_hist = zeros(1, N+1);   y_meas_hist(1) = y_hist(1);


%% Main Loop
fig = figure('units','normalized','outerposition',[0 0 1 1],'color','w');
for i = 1:N

    %% KF Observer: Predict
    x_hat_pred = A * x_hat + b * U(1);
    P_obs_pred = A * P_obs * A' + q_sys * eye(2);
    y_meas = c' * x_true + sqrt(r_sys)*randn; % Noisy Measurement

    %% KF Observer: Update
    Inn_obs = y_meas - c' * x_hat_pred;
    S_obs   = c' * P_obs_pred * c + r_sys;
    K_obs   = P_obs_pred * c / S_obs;
    x_hat   = x_hat_pred + K_obs * Inn_obs;
    P_obs   = (eye(2) - K_obs*c') * P_obs_pred;

    %% MPC Prediction (using x̂, not x_true)
    yhat = Z * x_hat + M * U;
    e    = yref(i:i+Ky-1) - yhat;

    %% MPC Kalman Update (U optimizer)
    P_pred = P + Q_mpc;
    S      = H * P_pred * H' + r * eye(Ky);
    K      = P_pred * H' / S;
    U      = U + K * e;
    P      = (I_k - K*H) * P_pred * (I_k - K*H)' + K*(r*eye(Ky))*K';

    %% Input Constraints
    U = max(min(U, umax), umin);
    if i > 1
        mu_val = min(deltaumax / max(abs(diff([u_hist(i-1); U]))), 1);
        U      = U * mu_val;
    end

    %% Apply to Real System (with process noise)
    u_apply  = U(1);
    x_true   = A * x_true + b * u_apply + sqrt(q_sys)*randn(2,1);

    %% Store
    x_true_hist(:, i+1) = x_true;
    x_hat_hist(:,  i+1) = x_hat;
    u_hist(i)            = u_apply;
    y_hist(i+1)          = c' * x_true;
    y_meas_hist(i+1)     = y_meas;
    U                    = [U(2:end); U(end)];

    %% Plot Update
    if mod(i, plot_interval) == 0 || i == N
        t_pred = t_hist(i+1 : min(i+Ky, N+1));

        subplot(221); hold on; grid on;
        plot(t_hist(1:N), yref(1:N), 'r', 'LineWidth', 1.5)
        % scatter(t_hist(1:i+1), yhat(1:length(t_pred)), 'g','filled');
        plot(t_hist(1:i+1), y_hist(1:i+1), 'b',   'LineWidth', 1.5);
        % plot(t_pred, y_meas_hist(1:i+1), 'k--', 'LineWidth', 1.0);
        xlabel('t [s]'); ylabel('y')
        title(sprintf('KF-MPC  Ky=%d  Ku=%d  q=%.e  r=%.e', Ky, Ku, q, r))
        legend('y_{ref}', 'y_{meas}', 'y[n]', 'prediction')

        subplot(222); hold on; grid on;
        yline(0,'k--')
        plot(t_hist(1:i), yref(1:i)' - y_hist(1:i), 'b', 'LineWidth', 1.5);
        xlabel('t [s]'); title('Tracking Error')

        subplot(223); hold on; grid on;
        yline(umax,'r--'); yline(umin,'r--')

        plot(t_hist(1:i), u_hist(1:i), 'b', 'LineWidth', 1.5);
        xlabel('t [s]'); ylabel('u'); title('Control Input')

        subplot(224); hold on; grid on;
        plot(t_hist(1:i+1), x_true_hist(1,1:i+1), 'k-',  'LineWidth', 1.0);
        plot(t_hist(1:i+1), x_hat_hist(1,1:i+1), 'r--', 'LineWidth', 1.5);
        plot(t_hist(1:i+1), x_true_hist(2,1:i+1), 'k-',  'LineWidth', 1.0);
        plot(t_hist(1:i+1), x_hat_hist(2,1:i+1), 'r--', 'LineWidth', 1.5);
        xlabel('t [s]'); ylabel('x_1'); title('State: true vs estimated')
        legend('x_1 true','x_1 estimated','x_2 true','x_2 estimated')

        drawnow
    end

end


%%  Utility Functions
function [M, Z] = mpc_matrices(A, b, c, Ku, Ky)
Z = zeros(Ky, length(b));
for k = 1:Ky
    Z(k,:) = c' * A^k;
end
M = zeros(Ky, Ku+1);
for i = 1:Ky
    M(i,1) = c' * A^(i-1) * b;
    for j = 2:min(i, Ku+1)
        M(i,j) = M(i-1, j-1);
    end
end
for k = Ku+2:Ky
    acc = 0;
    for l = 1:k-Ku-1
        acc = acc + c' * A^(l-1) * b;
    end
    M(k, Ku+1) = M(k, Ku+1) + acc;
end
end