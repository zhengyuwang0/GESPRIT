%% test coherence
clear; clc;
N_list = 50:50:400;
nb_Loop = 2000;
sigma2 = 1;
rrlist = [0,0.2,0.6];
k = 4;       
theta_true = zeros(1,k);

ESPRIT_MSE = zeros(nb_Loop,1);
MUSIC_MSE = zeros(nb_Loop,1);
GMUSIC_MSE = zeros(nb_Loop,1);
GESPRIT_MSE = zeros(nb_Loop,1);
CRB_MC = zeros(nb_Loop,1);
CRB_Theory = zeros(length(rrlist),length(N_list));
ESPRIT_MSE_E  = zeros(length(rrlist),length(N_list));
MUSIC_MSE_E  = zeros(length(rrlist),length(N_list));
GMUSIC_MSE_E  = zeros(length(rrlist),length(N_list));
GESPRIT_MSE_E  = zeros(length(rrlist),length(N_list));

for it = 1:length(N_list)
    N = N_list(it);
    T  = 2*N;
    c = N/T;
    theta_true = [-0.3*2*pi/N,0,0.2*2*pi/N,0.5*2*pi/N];
    theta_true1 = todeg(theta_true);
    delta = floor(N/3);
    n = N-delta;
    clear i
    a = @(theta) exp(1i*theta*(0:N-1)')/sqrt(N);
    diffa = @(theta) exp(1i*theta*(0:N-1)') *1i .*(0:N-1).' /sqrt(N);
    A = [];
    diffA = [];
    for tmp_index=1:length(theta_true)
        A = [A a(theta_true(tmp_index))];
        diffA = [diffA diffa(theta_true(tmp_index))];
    end
    Df = diffA;


    for ii = 1:1:length(rrlist)
        rr = rrlist(ii);
        P = (1-rr)*eye(k) + rr*ones(k);
        P = 25000*P;
        [V,D] = eig(A*P*A','vector');
        [D, index1] = sort(D,'descend');
        rho = real(D(1:k).')/sigma2;  % > sqrt(c)
        gg_true = (1-c*rho.^(-2))./(1+c*rho.^(-1));   % 2*1
        kk_true = n/N*(c+c*rho.^(-1))./(c+rho);

        for jj = 1 : nb_Loop
            S = sqrt(1/2) *sqrtm(P)*(randn(k,T) + 1i *randn(k,T));
            Z = sqrt(sigma2/2) * (randn(N,T) + 1i* randn(N,T));
            X = A*S + Z;     % N*T
            SCM = X*(X')/T;
            [U,eigs_SCM] = eig(SCM,'vector');
            [eigs_SCM, index] = sort(eigs_SCM,'descend');
            U = U(:, index);
            U_S = U(:,1:k);
    
            lambda_bar = eigs_SCM(1:k);
            sigma2_estim = mean(eigs_SCM(k+1:end));
            if lambda_bar>=(1+sqrt(c))^2
                ell_estim = (lambda_bar/sigma2_estim-(1+c))/2 + sqrt((lambda_bar/sigma2_estim-(1+c)).^2 - 4*c)/2;
                gg = (1-c*ell_estim.^(-2))./(1+c*ell_estim.^(-1));  
                kk = 1-gg;
            else
                gg = ones(k,1); kk = zeros(k,1);
            end
            % 
            ESPRIT_Theta = GetESPRITE(U_S,n,delta);   
            GESPRIT_Theta = GetGESPRITE(U_S,n,delta,gg,k);  
            ESPRIT_Theta = todeg(ESPRIT_Theta);
            GESPRIT_Theta = todeg(GESPRIT_Theta);
            ESPRIT_MSE(jj) =sum((ESPRIT_Theta.' - theta_true1).^2)/k;
            GESPRIT_MSE(jj) =sum((GESPRIT_Theta.' - theta_true1).^2)/k;

            [CRB, J_phi, J_theta] = crb_doa_stochastic_fromA(A, Df, P, sigma2, T, deg2rad(theta_true1));
            CRB_MC(jj) = trace(CRB)/k;
    
        end
        ESPRIT_MSE_E(ii,it)  = mean(ESPRIT_MSE,1);
        GESPRIT_MSE_E(ii,it)  = mean(GESPRIT_MSE,1);
        CRB_Theory(ii,it)    = mean(CRB_MC);
    end
end

rad2deg2 = (180/pi)^2;
CRB_Theory     = CRB_Theory     * rad2deg2;

figure();
hold on;
plot(N_list,log10(ESPRIT_MSE_E(1,:)),'LineStyle','-','Color','b','Marker','o','LineWidth',1.5);
plot(N_list,log10(ESPRIT_MSE_E(2,:)),'LineStyle','-','Color','b','Marker','+','LineWidth',1.5);
plot(N_list,log10(ESPRIT_MSE_E(3,:)),'LineStyle','-','Color','b','Marker','*','LineWidth',1.5);
plot(N_list,log10(GESPRIT_MSE_E(1,:)),'LineStyle','-','Color','r','Marker','o','LineWidth',1.5);
plot(N_list,log10(GESPRIT_MSE_E(2,:)),'LineStyle','-','Color','r','Marker','+','LineWidth',1.5);
plot(N_list,log10(GESPRIT_MSE_E(3,:)),'LineStyle','-','Color','r','Marker','*','LineWidth',1.5);
plot(N_list,log10(CRB_Theory(1,:)),'LineStyle','--','Color','#77AC30','LineWidth',1.5);
plot(N_list,log10(CRB_Theory(2,:)),'LineStyle','--','Color','#77AC30','LineWidth',1.5);
plot(N_list,log10(CRB_Theory(3,:)),'LineStyle','--','Color','#77AC30','LineWidth',1.5);
hold off;
legend('ESPRIT-1','ESPRIT-2','ESPRIT-3','GESPRIT-1','GESPRIT-2','GESPRIT-3','CRB-1','CRB-2','CRB-3')


function [CRB_phi, J_phi, J_theta] = crb_doa_stochastic_fromA(A, diffA, P, sigma2, T, phi_true)
    [N, k] = size(A);
    P = (P + P')/2;
    Rx = A * P * A' + sigma2 * eye(N);
    Rx_inv = inv(Rx);
    dR_theta = cell(k,1);
    for i = 1:k
        ai_diff = diffA(:, i);              % N x 1, da/dtheta_i
        term1 = ai_diff * (P(i, :) * A');   % N x N
        term2 = A * (P(:, i) * ai_diff');   % N x N   (注意是 ai_diff', 共轭转置在上层已处理)
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
    J_theta = (J_theta + J_theta')/2;   % 数值对称化
    phi_true = phi_true(:);                 
    dtheta_dphi = pi * cos(phi_true);          % k x 1
    D = diag(dtheta_dphi);                     % k x k
    J_phi = D * J_theta * D;
    J_phi = (J_phi + J_phi')/2;                % 数值对称
    CRB_phi = inv(J_phi);
end
