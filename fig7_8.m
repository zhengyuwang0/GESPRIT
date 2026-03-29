clear; clc;
N = 400;
T  = 800;
c = N/T;   
nb_Loop = 50;
testcase = 'widely';  % or closely

switch testcase
    case 'widely'
        theta_true = [-0.6*pi,-pi*0.22,pi/4,pi*2/5];
        theta_true1 = todeg(theta_true);
        k = length(theta_true);
        n = N-1;
        delta = 1;
        rr = 0.2;
        P1 = (1-rr)*eye(k) + rr*ones(k);
        P1 = 2*P1;
    case 'closely'
        theta_true = [-0.3*2*pi/N,0,0.2*2*pi/N,0.5*2*pi/N];
        theta_true1 = todeg(theta_true);
        k = length(theta_true);
        delta = 120;
        n = N-delta;
        rr = 0.2;
        P = (1-rr)*eye(k) + rr*ones(k);
        P1 = 2000*P;
end

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

[V,E] = eig(A*P1*A','vector');
[E, index1] = sort(E,'descend');
SNRList = -4:4:30;
powList = db2pow(SNRList)/E(k);  
sigma2 = 1;
SNR_cond = pow2db(sqrt(c)/E(k))


ESPRIT_MSE = zeros(nb_Loop,length(SNRList));
MUSIC_MSE = zeros(nb_Loop,length(SNRList));
GMUSIC_MSE = zeros(nb_Loop,length(SNRList));
GESPRIT_MSE = zeros(nb_Loop,length(SNRList));

CRB_Theory = zeros(1,length(SNRList));

for it=1:1:length(SNRList)
    pow = powList(it);
    P = pow*P1;

    [V,D] = eig(A*P*A','vector');
    [D, index1] = sort(D,'descend');
    V = V(:, index1);
    V = V(:,1:k);
    rho = D(1:k)/sigma2;
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
        if lambda_bar>=(1+sqrt(c))^2
            ell_estim = (lambda_bar-(1+c))/2 + sqrt((lambda_bar-(1+c)).^2 - 4*c)/2;
            gg = (1-c*ell_estim.^(-2))./(1+c*ell_estim.^(-1));  
            kk = 1-gg;
        else
            gg = ones(k,1); kk = zeros(k,1);
        end

        % MUSIC_Theta = GetMusic(U_S);
        % GMUSIC_Theta = GetGMusic(U_S,eigs_SCM,c);
        % MUSIC_Theta = todeg(MUSIC_Theta);
        % GMUSIC_Theta = todeg(GMUSIC_Theta);
        % MUSIC_MSE(jj,it) = sum((MUSIC_Theta - theta_true1).^2)/k;
        % GMUSIC_MSE(jj,it) = sum((GMUSIC_Theta - theta_true1).^2)/k;
        ESPRIT_Theta = GetESPRITE(U_S,n,delta); 
        GESPRIT_Theta = GetGESPRITE(U_S,n,delta,gg_true,k); 
        ESPRIT_Theta = todeg(ESPRIT_Theta);
        GESPRIT_Theta = todeg(GESPRIT_Theta);
        ESPRIT_MSE(jj,it) =sum((ESPRIT_Theta.' - theta_true1).^2)/k;
        GESPRIT_MSE(jj,it) =sum((GESPRIT_Theta.' - theta_true1).^2)/k;

    end
    ESPRIT_MSE_E  = mean(ESPRIT_MSE,1);
    GESPRIT_MSE_E  = mean(GESPRIT_MSE,1);
    % MUSIC_MSE_E  = mean(MUSIC_MSE,1);
    % GMUSIC_MSE_E  = mean(GMUSIC_MSE,1);
    [CRB, J_phi, J_theta] = crb_doa_stochastic_fromA(A, Df, P, sigma2, T, deg2rad(theta_true1));
    CRB_Theory(it) = trace(CRB)/k;
end

rad2deg2 = (180/pi)^2;
CRB_Theory = CRB_Theory * rad2deg2;

figure();
hold on;
plot(SNRList,log10(ESPRIT_MSE_E),'LineStyle','-','Color','#FF6100','Marker','*','LineWidth',1.5);
plot(SNRList,log10(GESPRIT_MSE_E.'),'LineStyle','-','Color','#A020F0','Marker','^','LineWidth',1.5);
plot(SNRList,log10(CRB_Theory),'LineStyle','--','Color','#77AC30','Marker','o','LineWidth',1.5);
hold off;
legend('ESPRIT MSE','GESPRIT MSE','CRB')
% legend('MUSIC MSE','GMUSIC MSE','ESPRIT MSE','GESPRIT MSE','CRB')
xlabel('SNR')
ylabel('log10')
xlim([SNRList(1) SNRList(end)])



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

