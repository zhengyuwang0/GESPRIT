close all; clear; clc;
sigma2 = 1;
CNT    = 2000;          
N_list = 50:50:400;

% Case 1: widely-spaced DoAs
caseWide.name = 'widely-spaced';
caseWide.theta_fun = @(N) [0, pi/4];
caseWide.P = 2*[3 1.2; 1.2 3];   
caseWide.ndelta_fun = @(N,theta_true) deal(N-1, 1);

% Case 2: closely-spaced DoAs
caseClose.name = 'closely-spaced';
caseClose.theta_fun = @(N) [0, 0.8*2*pi/N];
caseClose.P = 5*eye(2);    

caseClose.ndelta_fun = @(N,theta_true) deal(floor(2*N/3), floor(N/3));
% caseClose.ndelta_fun = @(N,theta_true) deal(N-round(pi/abs(theta_true(2))), round(pi/abs(theta_true(2))));

%  Run simulation
resWide  = run_one_case(N_list, CNT, sigma2, caseWide);
resClose = run_one_case(N_list, CNT, sigma2, caseClose);

%  Power-law fitting
[fitWide1, txtWide1]   = power_fit(N_list, resWide.mean_err1);
[fitWide2, txtWide2]   = power_fit(N_list, resWide.mean_err2);
[fitClose1, txtClose1] = power_fit(N_list, resClose.mean_err1);
[fitClose2, txtClose2] = power_fit(N_list, resClose.mean_err2);

%  Plot
% ==========================
figure('Color','w','Position',[100,80,1000,760]);
tl = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
% ---- (1) widely-spaced, ||Phi_hat - Phi_bar||
ax1 = nexttile;
plot_one_panel(ax1, N_list, resWide.mean_err1, resWide.std_err1, fitWide1, ...
    '$\|\hat{\Phi}-\bar{\Phi}\|$', txtWide1, true);
ylabel('Spectral norm errors','Interpreter','latex');
% ---- (2) widely-spaced, ||Phi_hat^G - Phi||
ax2 = nexttile;
plot_one_panel(ax2, N_list, resWide.mean_err2, resWide.std_err2, fitWide2, ...
    '$\|\hat{\Phi}^{G}-\Phi\|$', txtWide2, true);
% ---- (3) closely-spaced, ||Phi_hat - Phi_bar||
ax3 = nexttile;
plot_one_panel(ax3, N_list, resClose.mean_err1, resClose.std_err1, fitClose1, ...
    '$\|\hat{\Phi}-\bar{\Phi}\|$', txtClose1, false);
ylabel('Spectral norm errors','Interpreter','latex');
% ---- (4) closely-spaced, ||Phi_hat^G - Phi||
ax4 = nexttile;
plot_one_panel(ax4, N_list, resClose.mean_err2, resClose.std_err2, fitClose2, ...
    '$\|\hat{\Phi}^{G}-\Phi\|$', txtClose2, false);


%  Local functions
% ==========================
function res = run_one_case(N_list, CNT, sigma2, cfg)
    numN = length(N_list);
    err1 = zeros(CNT, numN);   % ||Phi_hat - Phi_bar||_2
    err2 = zeros(CNT, numN);   % ||Phi_hat^G - Phi_true||_2

    for it = 1:numN
        N = N_list(it);
        T = 2*N;
        c = N/T;
        theta_true = cfg.theta_fun(N);
        P = cfg.P;
        k = length(theta_true);
        [n, delta] = cfg.ndelta_fun(N, theta_true);

        % Selection matrices
        J = eye(N);
        J1 = J(1:n,:);
        J2 = J(1+delta:n+delta,:);

        % Steering matrix
        A = build_steering(theta_true, N);

        % Population signal subspace
        [V, D] = eig(A*P*A', 'vector');
        [D, index1] = sort(real(D), 'descend');
        V = V(:, index1);
        V = V(:,1:k);
        rho = D(1:k)/sigma2;
        gg_true = (1-c*rho.^(-2))./(1+c*rho.^(-1));   
        kk_true = n/N*(c+c*rho.^(-1))./(c+rho);

        % True Phi
        Phi1_true = V'*(J1'*J1)*V;
        Phi2_true = V'*(J1'*J2)*V;
        Phi_true  = Phi1_true \ Phi2_true;

        % ===== Build Phi_bar =====
        Gmat = sqrt(gg_true(:) * gg_true(:).');     % [g_i g_j]^(1/2)
        Phi1_bar = Phi1_true .* Gmat + diag(kk_true(:));
        Phi2_bar = Phi2_true .* Gmat;
        Phi_bar  = Phi1_bar \ Phi2_bar;

        for cnt = 1:CNT
            % ===== Data generation =====
            S = sqrtm(P) * randn(k, T);
            Z = sqrt(sigma2/2) * (randn(N,T) + 1i*randn(N,T));
            X = A*S + Z;

            SCM = (X * X') / T;
            SCM = (SCM + SCM') / 2; 

            [U, eigs_SCM] = eig(SCM, 'vector');
            [eigs_SCM, idx2] = sort(real(eigs_SCM), 'descend');
            U = U(:, idx2);
            U_S = U(:, 1:k);

            % ===== Phi_hat =====
            Phi1_hat = U_S'*(J1'*J1)*U_S;
            Phi2_hat = U_S'*(J1'*J2)*U_S;
            Phi_hat  = Phi1_hat \ Phi2_hat;

            % ===== Estimate ell, gg, kk =====
            lambda_bar = eigs_SCM(1:k);
            if lambda_bar>=(1+sqrt(c))^2
                ell_estim = (lambda_bar-(1+c))/2 + sqrt((lambda_bar-(1+c)).^2 - 4*c)/2;
            end
            gg = (1-c*ell_estim.^(-2))./(1+c*ell_estim.^(-1));  
            kk = n/N*(c+c*ell_estim.^(-1))./(c+ell_estim);
  
            % ===== Build debiased Phi^G =====
            g = diag((real(gg)).^(-1/2));  % k*K
            Tilde_Phi1 = g*(Phi1_hat-n/N*eye(k,k))*g + n/N*eye(k,k);
            Tilde_Phi2 = g*Phi2_hat*g;
            Tilde_Phi = Tilde_Phi1\Tilde_Phi2;

            % ===== Errors =====
            e1 = eig(Phi_hat);  e2 = eig(Phi_bar);
            [~,i1] = sort(real(e1)); [~,i2] = sort(real(e2));
            e1 = e1(i1); e2 = e2(i2);
            err1(cnt,it) = abs(e1(end)-e2(end));

            e3 = eig(Tilde_Phi); e4 = eig(Phi_true);
            [~,i3] = sort(real(e3)); [~,i4] = sort(real(e4));
            e3 = e3(i3); e4 = e4(i4);
            err2(cnt,it) = abs(e3(end)-e4(end));

        end
    end

    res.err1 = err1;
    res.err2 = err2;
    res.mean_err1 = mean(err1, 1);
    res.std_err1  = std(err1, 0, 1);
    res.mean_err2 = mean(err2, 1);
    res.std_err2  = std(err2, 0, 1);
end

function A = build_steering(theta_true, N)
    a = @(theta) exp(1i * theta * (0:N-1)') / sqrt(N);
    k = length(theta_true);
    A = zeros(N, k);
    for ii = 1:k
        A(:,ii) = a(theta_true(ii));
    end
end

function [yfit, txt] = power_fit(x, y)
    p = polyfit(log(x(:)), log(y(:)), 1);
    b = p(1);
    a = exp(p(2));
    yfit = a * x.^b;
    txt = sprintf('%.2fN^{%.2f}', a, b);
end

function plot_one_panel(ax, x, y, ystd, yfit, leg1, leg2, sciTop)
    axes(ax); hold on;
    errorbar(x, y, ystd, 'LineStyle','none', ...
        'Color',[0.6 0.6 0.6], 'LineWidth',1.0, 'CapSize',4);
    h1 = plot(x, y, 'o', ...
        'Color',[0.2 0.4 1.0], ...
        'MarkerFaceColor','none', ...
        'MarkerSize',6, ...
        'LineWidth',1.2);

    h2 = plot(x, yfit, '-', ...
        'Color',[0.85 0.35 0.35], ...
        'LineWidth',2.0);

    grid on; box on;
    xlabel('$N$','Interpreter','latex');

    legend([h1,h2], {leg1, ['$' leg2 '$']}, ...
        'Interpreter','latex', ...
        'Location','northeast');

    ax.FontSize = 12;
    ax.LineWidth = 1.0;

    if sciTop
        ax.YAxis.Exponent = -2;
    end
end