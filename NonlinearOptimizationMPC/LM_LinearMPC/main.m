clc, clear all, close all, warning off;
%% Linear Discrete Time Model Predictive Control

A = [1.1 0.6; 0.2 0.1];  % System matrix
b = [1; 0];               % Input matrix
c = [1 1]';               % Output matrix

%% Algorithm Parameters
Ts = 1e-1;          % Sampling time [s]
T  = 40;            % Total simulation time [s]
N  = round(T / Ts); % Number of simulation steps

umin      = -1;     % Input lower bound        (hard saturation limit)
umax      =  1;     % Input upper bound        (hard saturation limit)
Ku        = 5;      % Control horizon          (number of free input moves per step, Ku <= Ky)
Ky        = 10;     % Prediction horizon       (number of future output steps considered)
deltaumax = 0.2;    % Max input rate           (maximum allowed |u[k] - u[k-1]| per step)
lamda     = 0.5;    % Delta-u penalty weight   (larger = smoother input, slower tracking)

plot_interval = 10; % refresh rate  

%% Time and Reference Signal  first half: multi-step, second half: sinusoidal
t_hist   = (0:N)' * Ts;
t_ref    = (0:N+Ky-2)' * Ts;
half     = floor(length(t_ref) / 2);
seg      = floor(half / 4);
ref_step = [0.4*ones(seg,1); 0.8*ones(seg,1); 1.2*ones(seg,1); 0.8*ones(half-3*seg,1)];
ref_sin  = 0.3*sin(0.5*pi*t_ref(1:length(t_ref)-half)) + 0.8;
yref     = [ref_step; ref_sin];

%% Initial Conditions
x = zeros(2,   N+1);    x(:,1) = [0; 0];
u = zeros(Ku+1, N+1);   u(:,1) = zeros(Ku+1, 1);
y = zeros(1,   N+1);    y(1)   = c' * x(:,1);

%% MPC Matrices — computed once offline
[L, M, Z] = LDTS_MPC_matrices(A, b, c, Ku, Ky);
Hes   = M'*M + lamda*L;
Hters = inv(Hes);

%% Main Loop
figure('units','normalized','outerposition',[0 0 1 1],'color','w')
for i = 1:N

    % Gradient and Newton step
    g      = Hes*u(:,i) - (M'*(yref(i:i+Ky-1) - Z*x(:,i)) + [lamda*u(1,i); zeros(Ku,1)]);
    deltau = -Hters * g;

    % delta-u rate constraint
    mu     = min(1, deltaumax / max([max(abs(deltau)), max(abs(diff(u(:,i) + deltau)))]));
    deltau = mu * deltau;

    % Update and saturate
    u(:,i+1) = u(:,i) + deltau;
    u(1,i+1) = max(umin, min(umax, u(1,i+1)));

    % System update
    x(:,i+1) = A*x(:,i) + b*u(1,i+1);
    y(i+1)   = c'*x(:,i+1);

    % Plot
    if mod(i, plot_interval) == 0 || i == N
        clf

        subplot(221); hold on; grid on;
        plot(t_hist(1:N), yref(1:N),  'r', 'LineWidth', 2)
        plot(t_hist(1:i+1), y(1:i+1), 'k', 'LineWidth', 2)
        xlabel('t [s]'); ylabel('y')
        title(sprintf('Ky=%d | Ku=%d | \\lambda=%.2f', Ky, Ku, lamda))
        legend('y_{ref}', 'y[n]')
        axis([0 T min(yref)-.5 max(yref)+.5])

        subplot(222); hold on; grid on;
        plot(t_hist(1:i+1), yref(1:i+1)' - y(1:i+1), 'b', 'LineWidth', 1)
        yline(0, 'k--')
        xlabel('t [s]'); title('Tracking Error')

        subplot(223); hold on; grid on;
        plot(t_hist(1:i+1), u(1,1:i+1), 'b', 'LineWidth', 1)
        yline(umax, 'r--', 'u_{max}'); yline(umin, 'r--', 'u_{min}')
        xlabel('t [s]'); ylabel('u'); title('Input')

        subplot(224); hold on; grid on;
        plot(t_hist(1:i+1), x(1,1:i+1), 'k--', 'LineWidth', 1)
        plot(t_hist(1:i+1), x(2,1:i+1), 'r--', 'LineWidth', 1)
        xlabel('t [s]'); ylabel('x_i'); title('State')
        legend('x_1', 'x_2')

        drawnow
        
    end
end

%% -----------------------------------------------------------------------
function [L, M, Z] = LDTS_MPC_matrices(A, b, c, Ku, Ky)

% L — tridiagonal delta-u penalty matrix (Ku+1 x Ku+1)
L = 2*eye(Ku+1);
L(end,end) = 1;
for i = 1:Ku
    L(i+1,i) = -1;
    L(i,i+1) = -1;
end
if Ku == 0, L = 1; end

% Z — free response (Ky x nx)
Z = zeros(Ky, length(b));
for k = 1:Ky
    Z(k,:) = c'*A^k;
end

% M — forced response (Ky x Ku+1)
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