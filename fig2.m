%% Fig. 2: two panels in one run
clear; clc; close all;
N       = 400;
T       = 800;
c       = N/T;
sigma2  = 1;
nb_Loop = 100;     
k       = 3;
P       = 2*eye(k);  

% (a) Left panel: MSE versus n, with Delta = 1
theta_left = [0, pi/4, 2*pi/3];
delta_left = 1;
n_list     = 4:2:(N-2);  

[mseG_left, mseE_left] = run_mse_vs_n(theta_left, P, N, T, sigma2, nb_Loop, n_list, delta_left,c);

% (b) Right panel: MSE versus Delta, with n = N - Delta
theta_right = [0, 0.8*2*pi/N, 1.6*2*pi/N];
ds          = 4:2:(N-2);  

[mseG_right, mseE_right] = run_mse_vs_delta(theta_right, P, N, T, sigma2, nb_Loop, ds,c);

delta_ref = abs(pi/theta_right(3));

figure('Color','w','Position',[100,100,1050,420]);
tl = tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

% -------- left subplot --------
nexttile;
h1 = semilogy(n_list, mseG_left, 'o-', 'LineWidth', 1.2, 'MarkerSize', 6); hold on;
h2 = semilogy(n_list, mseE_left, 'o-', 'LineWidth', 1.2, 'MarkerSize', 6);
grid on; box on;
xlabel('$n$','Interpreter','latex');
ylabel('MSE (in deg$^2$)','Interpreter','latex');
legend([h1,h2], {'G-ESPRIT','ESPRIT'}, ...
    'Interpreter','latex','Location','northeast');
title('(a)','Interpreter','latex');
xlim([min(n_list), max(n_list)]);

% -------- right subplot --------
nexttile;
h3 = semilogy(ds, mseG_right, 'o-', 'LineWidth', 1.2, 'MarkerSize', 6); hold on;
h4 = semilogy(ds, mseE_right, 'o-', 'LineWidth', 1.2, 'MarkerSize', 6);
h5 = xline(delta_ref, 'k--', 'LineWidth', 1.2);
grid on; box on;
xlabel('$\Delta$','Interpreter','latex');
ylabel('MSE (in deg$^2$)','Interpreter','latex');
legend([h3,h4,h5], {'G-ESPRIT','ESPRIT','$|\pi/\bar{\theta}_3|$'}, ...
    'Interpreter','latex','Location','northwest');
title('(b)','Interpreter','latex');
xlim([min(ds), max(ds)]);

% =======================
% Local functions
% =======================

function [mseG_mean, mseE_mean] = run_mse_vs_n(theta_true, P, N, T, sigma2, nb_Loop, n_list, delta,c)
    k = length(theta_true);
    A = build_steering_matrix(theta_true, N);

    GESPRIT_MSE = zeros(nb_Loop, length(n_list));
    ESPRIT_MSE  = zeros(nb_Loop, length(n_list));

    for it = 1:length(n_list)
        n = n_list(it);
        for jj = 1:nb_Loop
            [thetaG, thetaE] = one_trial_estimation(A, P, N, T, sigma2, k, n, delta,c);
            GESPRIT_MSE(jj,it) = sum((todeg(thetaG.') - todeg(theta_true)).^2)/k;
            ESPRIT_MSE(jj,it)  = sum((todeg(thetaE.') - todeg(theta_true)).^2)/k;
        end
    end

    mseG_mean = mean(GESPRIT_MSE, 1);
    mseE_mean = mean(ESPRIT_MSE, 1);
end

function [mseG_mean, mseE_mean] = run_mse_vs_delta(theta_true, P, N, T, sigma2, nb_Loop, ds,c)
    k = length(theta_true);
    A = build_steering_matrix(theta_true, N);

    GESPRIT_MSE = zeros(nb_Loop, length(ds));
    ESPRIT_MSE  = zeros(nb_Loop, length(ds));

    for it = 1:length(ds)
        delta = ds(it);
        n = N - delta;

        for jj = 1:nb_Loop
            [thetaG, thetaE] = one_trial_estimation(A, P, N, T, sigma2, k, n, delta,c);
            GESPRIT_MSE(jj,it) = sum((todeg(thetaG.') - todeg(theta_true)).^2)/k;
            ESPRIT_MSE(jj,it)  = sum((todeg(thetaE.') - todeg(theta_true)).^2)/k;
        end
    end
    mseG_mean = mean(GESPRIT_MSE, 1);
    mseE_mean = mean(ESPRIT_MSE, 1);
end

function [thetaG, thetaE] = one_trial_estimation(A, P, N, T, sigma2, k, n, delta,c)
    S = sqrt(1/2) *sqrtm(P)*(randn(k,T) + 1i *randn(k,T));
    Z = sqrt(sigma2/2) * (randn(N,T) + 1i*randn(N,T));
    X = A*S + Z;
    SCM = (X * X') / T;
    SCM = (SCM + SCM') / 2;   
    [U,eigs_SCM] = eig(SCM,'vector');
    [eigs_SCM, index] = sort(eigs_SCM,'descend');
    U = U(:,index);
    U_S = U(:,1:k);
    lambda_bar = eigs_SCM(1:k);
    if lambda_bar>=(1+sqrt(c))^2
        ell_estim = (lambda_bar-(1+c))/2 + sqrt((lambda_bar-(1+c)).^2 - 4*c)/2;
        gg = (1-c*ell_estim.^(-2))./(1+c*ell_estim.^(-1));  
    else
        gg = ones(k,1); 
    end
    % Estimation
    thetaG = GetGESPRITE(U_S, n, delta, gg, k);
    thetaE = GetESPRITE(U_S, n, delta);
    thetaG = sort(mod(real(thetaG(:)), 2*pi));
    thetaE = sort(mod(real(thetaE(:)), 2*pi));
end

function A = build_steering_matrix(theta_true, N)
    a = @(theta) exp(1i * theta * (0:N-1)') / sqrt(N);
    k = length(theta_true);
    A = zeros(N, k);
    for ii = 1:k
        A(:,ii) = a(theta_true(ii));
    end
end

function ansangle = todeg(theta)
    ansangle = rad2deg(asin(theta/pi));
end

