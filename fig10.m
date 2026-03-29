clear; clc; close all;
N = 400;
T = 800;
c = N/T;
nb_Loop = 20;   

theta_true = [-0.3*2*pi/N, 0, 0.2*2*pi/N, 0.5*2*pi/N];
theta_true_deg = rad2deg(theta_true);
k = length(theta_true);

delta = 120;
n = N - delta;

rr = 0.2;
P0 = (1-rr)*eye(k) + rr*ones(k);
P1 = 10 * P0;

sigma2 = 1;
SNRList = -4:4:28;

% Steering matrix
% =========================
a = @(theta) exp(1i*theta*(0:N-1)')/sqrt(N);
diffa = @(theta) exp(1i*theta*(0:N-1)') .* (1i*(0:N-1)') / sqrt(N);

A = zeros(N,k);
diffA = zeros(N,k);
for idx = 1:k
    A(:,idx) = a(theta_true(idx));
    diffA(:,idx) = diffa(theta_true(idx));
end
Df = diffA;

[V,E] = eig(A*P1*A','vector');
[E, index1] = sort(real(E),'descend');
V = V(:, index1);
V = V(:,1:k);

powList = db2pow(SNRList) / E(k);

% Run 3 noise models
% =========================
res_rademacher = run_one_noise_model('rademacher', N, T, c, nb_Loop, ...
    A, Df, P1, sigma2, SNRList, powList, theta_true_deg, n, delta, k);

res_uniform = run_one_noise_model('uniform', N, T, c, nb_Loop, ...
    A, Df, P1, sigma2, SNRList, powList, theta_true_deg, n, delta, k);

res_student = run_one_noise_model('student', N, T, c, nb_Loop, ...
    A, Df, P1, sigma2, SNRList, powList, theta_true_deg, n, delta, k);

% Plot 3 subfigures
% =========================
figure('Color','w','Position',[100,100,1280,420]);
tl = tiledlayout(1,3,'TileSpacing','compact','Padding','compact');

% ---------- (a) Rademacher ----------
ax1 = nexttile;
[h1,h2,h3,h4,h5] = plot_one_panel(ax1, SNRList, res_rademacher, '(a)');
ylabel('MSE (in deg^2)','Interpreter','latex');

% ---------- (b) Uniform ----------
ax2 = nexttile;
plot_one_panel(ax2, SNRList, res_uniform, '(b)');

% ---------- (c) Student-t / heavy-tailed ----------
ax3 = nexttile;
plot_one_panel(ax3, SNRList, res_student, '(c)');

lgd = legend([h1,h2,h3,h4,h5], ...
    {'ESPRIT','G-ESPRIT','MUSIC','G-MUSIC','CRB'}, ...
    'Orientation','horizontal', ...
    'NumColumns',5, ...
    'Box','on');
lgd.Layout.Tile = 'north';


% Local functions
% =========================
function res = run_one_noise_model(noiseType, N, T, c, nb_Loop, ...
    A, Df, P1, sigma2, SNRList, powList, theta_true_deg, n, delta, k)

    ESPRIT_MSE  = zeros(nb_Loop, length(SNRList));
    GESPRIT_MSE = zeros(nb_Loop, length(SNRList));
    MUSIC_MSE   = zeros(nb_Loop, length(SNRList));
    GMUSIC_MSE  = zeros(nb_Loop, length(SNRList));
    CRB_Theory  = zeros(1, length(SNRList));

    for it = 1:length(SNRList)
        pow = powList(it);
        P = pow * P1;

        for jj = 1:nb_Loop
            % source
            S = sqrt(1/2) * sqrtm(P) * (randn(k,T) + 1i*randn(k,T));

            % noise
            Z = generate_noise(noiseType, N, T, sigma2);

            % received data
            X = A*S + Z;
            SCM = X*(X')/T;
            SCM = (SCM + SCM')/2;

            [U, eigs_SCM] = eig(SCM, 'vector');
            [eigs_SCM, index] = sort(real(eigs_SCM), 'descend');
            U = U(:, index);
            U_S = U(:,1:k);

            lambda_bar = eigs_SCM(1:k);
            if lambda_bar>=(1+sqrt(c))^2
                ell_estim = (lambda_bar-(1+c))/2 + sqrt((lambda_bar-(1+c)).^2 - 4*c)/2;
                gg = (1-c*ell_estim.^(-2))./(1+c*ell_estim.^(-1));  
            else
                gg = ones(k,1);
            end

            % 4 methods
            MUSIC_Theta = GetMusic(U_S);
            GMUSIC_Theta = GetGMusic(U_S, eigs_SCM, c);
            ESPRIT_Theta = GetESPRITE(U_S, n, delta);
            GESPRIT_Theta = GetGESPRITE(U_S, n, delta, gg, k);

            MUSIC_Theta   = sort(rad2deg(real(MUSIC_Theta(:))));
            GMUSIC_Theta  = sort(rad2deg(real(GMUSIC_Theta(:))));
            ESPRIT_Theta  = sort(rad2deg(real(ESPRIT_Theta(:))));
            GESPRIT_Theta = sort(rad2deg(real(GESPRIT_Theta(:))));
            theta_ref     = sort(theta_true_deg(:));

            MUSIC_MSE(jj,it)   = mean((MUSIC_Theta  - theta_ref).^2);
            GMUSIC_MSE(jj,it)  = mean((GMUSIC_Theta - theta_ref).^2);
            ESPRIT_MSE(jj,it)  = mean((ESPRIT_Theta - theta_ref).^2);
            GESPRIT_MSE(jj,it) = mean((GESPRIT_Theta - theta_ref).^2);
        end

        [CRB, ~, ~] = crb_doa_stochastic_fromA(A, Df, P, sigma2, T, deg2rad(theta_true_deg));
        CRB_Theory(it) = trace(CRB)/k;
    end

    res.ESPRIT_MSE_E  = mean(ESPRIT_MSE,1);
    res.GESPRIT_MSE_E = mean(GESPRIT_MSE,1);
    res.MUSIC_MSE_E   = mean(MUSIC_MSE,1);
    res.GMUSIC_MSE_E  = mean(GMUSIC_MSE,1);
    res.CRB_Theory    = CRB_Theory * (180/pi)^2;
end

function Z = generate_noise(noiseType, N, T, sigma2)
    switch lower(noiseType)
        case 'rademacher'
            W_real = sign(rand(N,T) - 0.5);
            W_imag = sign(rand(N,T) - 0.5);
            Z = sqrt(sigma2/2) * (W_real + 1i*W_imag);

        case 'uniform'
            a_uni = sqrt(3*sigma2/2);
            Zr = a_uni * (2*rand(N,T) - 1);
            Zi = a_uni * (2*rand(N,T) - 1);
            Z = Zr + 1i*Zi;

        case 'student'
            nu = 1.5;
            Zr = trnd(nu, N, T);
            Zi = trnd(nu, N, T);
            Z0 = Zr + 1i*Zi;

            Z0 = Z0 / sqrt(mean(abs(Z0(:)).^2)) * sqrt(sigma2);
            Z = Z0;

        otherwise
            error('Unknown noise type.');
    end
end

function ell_estim = estimate_ell(lambda_bar, c)
    threshold = (1 + sqrt(c))^2;
    ell_estim = zeros(size(lambda_bar));

    for ii = 1:length(lambda_bar)
        if lambda_bar(ii) >= threshold
            tmp = lambda_bar(ii) - (1 + c);
            disc = tmp^2 - 4*c;
            disc = max(disc, 0);
            ell_estim(ii) = (tmp + sqrt(disc))/2;
        else
            ell_estim(ii) = 1e6;   % 对应 gg ≈ 1
        end
    end
end

function [h1,h2,h3,h4,h5] = plot_one_panel(ax, SNRList, res, labelText)
    axes(ax); hold on; box on;

    h1 = semilogy(SNRList, res.ESPRIT_MSE_E,  '-x', 'LineWidth',1.6, 'MarkerSize',6);
    h2 = semilogy(SNRList, res.GESPRIT_MSE_E, '-o', 'LineWidth',1.6, 'MarkerSize',7);
    h3 = semilogy(SNRList, res.MUSIC_MSE_E,   '-*', 'LineWidth',1.6, 'MarkerSize',5);
    h4 = semilogy(SNRList, res.GMUSIC_MSE_E,  '-o', 'LineWidth',1.6, 'MarkerSize',7);
    h5 = semilogy(SNRList, res.CRB_Theory,    '--', 'LineWidth',1.6);

    grid on;
    ax.GridLineStyle = '--';
    ax.GridAlpha = 0.4;
    ax.FontSize = 14;
    ax.LineWidth = 1.0;

    xlabel('SNR $R_{\mathrm{SNR}}$ (dB)', 'Interpreter','latex');
    xlim([SNRList(1), SNRList(end)]);
    ylim([1e-7, 1]);

    title(labelText, 'Interpreter','latex', 'Y', -0.25);
end

function [CRB_phi, J_phi, J_theta] = crb_doa_stochastic_fromA(A, diffA, P, sigma2, T, phi_true)
    [N, k] = size(A);
    P = (P + P')/2;
    Rx = A * P * A' + sigma2 * eye(N);
    Rx_inv = inv(Rx);

    dR_theta = cell(k,1);
    for i = 1:k
        ai_diff = diffA(:, i);
        term1 = ai_diff * (P(i, :) * A');
        term2 = A * (P(:, i) * ai_diff');
        dR_theta{i} = term1 + term2;
        dR_theta{i} = (dR_theta{i} + dR_theta{i}')/2;
    end

    J_theta = zeros(k, k);
    for i = 1:k
        Ri = Rx_inv * dR_theta{i};
        for j = 1:k
            J_theta(i, j) = T * real(trace(Ri * Rx_inv * dR_theta{j}));
        end
    end
    J_theta = (J_theta + J_theta')/2;

    phi_true = phi_true(:);
    dtheta_dphi = pi * cos(phi_true);
    D = diag(dtheta_dphi);
    J_phi = D * J_theta * D;
    J_phi = (J_phi + J_phi')/2;
    CRB_phi = inv(J_phi);
end