%% subfig (d)-(e)
clear; clc
sigma2 = 1;
nb_Loop = 2000;

coeff = 50:50:400;
ESPRIT_MSE = zeros(nb_Loop,length(coeff));
GESPRIT_MSE = zeros(nb_Loop,length(coeff));
CRB_Theory = zeros(1,length(coeff));
ESPRIT_estimate = zeros(2,nb_Loop);
GESPRIT_estimate = zeros(2,nb_Loop);

for it = 1:1:length(coeff)
    N = coeff(it);
    T  = 2*N;
    c = N/T;
    testcase = 'widely';   % or closely

    switch testcase
        case 'widely'
            theta_true = [0,pi/4];
            theta_true1 = todeg(theta_true);
            n = N-1;
            delta = 1;
            % P = [3,1.2;1.2,3];   % uncorrelated sources, sub-fig (d)
            P = 2*[1,0;0,1];   % correlated sources, sub-fig (f)
        case 'closely'
            theta_true = [0,0.8*2*pi/N];
            theta_true1 = todeg(theta_true);
            n = floor(2*N/3);
            delta = floor(N/3);
            P = 2*[1,0;0,1];   % sub-fig (e)
    end
    k = length(theta_true);
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
    [V,D] = eig(A*P*A','vector');
    [D, index1] = sort(D,'descend');
    V = V(:, index1);
    V = V(:,1:k);
    rho = D(1:2)/sigma2;
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
        end
        gg = (1-c*ell_estim.^(-2))./(1+c*ell_estim.^(-1));  
        kk = n/N*(c+c*ell_estim.^(-1))./(c+ell_estim);

        ESPRIT_Theta = GetESPRITE(U_S,n,delta);   
        GESPRIT_Theta = GetGESPRITE(U_S,n,delta,gg,k);
        ESPRIT_Theta = todeg(ESPRIT_Theta);   
        GESPRIT_Theta = todeg(GESPRIT_Theta);  
        % 
        ESPRIT_MSE(jj,it) = sum((ESPRIT_Theta.' - theta_true1).^2) / k;
        ESPRIT_estimate(:,jj) = ESPRIT_Theta;

        GESPRIT_MSE(jj,it) = sum((GESPRIT_Theta.' - theta_true1).^2) / k;
        GESPRIT_estimate(:,jj) = GESPRIT_Theta;

    end
    ESPRIT_Bias_E(it) = sum((mean(ESPRIT_estimate,2)-theta_true1.').^2)/2;
    GESPRIT_Bias_E(it) = sum((mean(GESPRIT_estimate,2)-theta_true1.').^2)/2;
end

ESPRIT_MSE_E  = mean(ESPRIT_MSE,1);
ESPRIT_Var_E  = ESPRIT_MSE_E - ESPRIT_Bias_E;
GESPRIT_MSE_E  = mean(GESPRIT_MSE,1);
GESPRIT_Var_E  = GESPRIT_MSE_E - GESPRIT_Bias_E;

rad2deg2 = (180/pi)^2;

figure;
hold on;
plot(coeff,log10(GESPRIT_MSE_E));
plot(coeff,log10(GESPRIT_Var_E));
plot(coeff,log10(ESPRIT_MSE_E))
plot(coeff,log10(ESPRIT_Var_E))
plot(coeff,log10(CRB_Theory))
legend('GESPRIT-MSE','GESPRIT-VAR','ESPRIT-MSE','ESPRIT-VAR','CRB');
xlabel('N');
ylabel('MSE(dB)');


%% subfig (a)-(c)
clear; clc;
N_list = 20:30:200;
CNT = 50;
theta_hat = zeros(length(N_list),2);
theta_tilde = zeros(length(N_list),2);
theta_cc = zeros(length(N_list),2);
theta_bar = zeros(length(N_list),2);

for nn=1:1:length(N_list)
    N = N_list(nn);
    T  = 2*N;
    c = N/T;   
    testcase = 'closely'; 
    switch testcase
        case 'widely'
            theta_true = [0,pi/4];
            n = N-1;
            delta = 1;
             P = 0.8*[3,1.2;1.2,3];   % uncorrelated sources, sub-fig (a)
            % P = 2*[1,0;0,1];   % correlated sources, sub-fig (c)
        case 'closely'
            theta_true = [0,0.3*2*pi/N];
            n = floor(2*N/3);
            delta = floor(N/3);
            P = 2*[1,0;0,1];
    end
    theta_cc(nn,:) = theta_true;

    k = length(theta_true);
    clear i
    a = @(theta) exp(1i*theta*(0:N-1)')/sqrt(N);
    A = [];
    for tmp_index=1:length(theta_true)
        A = [A a(theta_true(tmp_index))];
    end
    [V,D] = eig(A*P*A','vector');
    [D, index1] = sort(D,'descend');
    V = V(:, index1);
    V = V(:,1:2);
    sigma2 = 1;
    rho = D(1:k)./sigma2;
    
    gg_true = (1-c*rho.^(-2))./(1+c*rho.^(-1));   % 2*1
    kk_true = n/N*(c+c*rho.^(-1))./(c+rho);
    
    % theta_bar
    J_tmp = eye(N);
    J1 = J_tmp(1:n,:);
    J2 = J_tmp(1+delta:n+delta,:);
    Phi1 =  V' * (J1' * J1) * V;
    Phi2 = V' * J1' * J2 * V;
    % ===== Build Phi_bar =====
    Gmat = sqrt(gg_true(:) * gg_true(:).');     % [g_i g_j]^(1/2)
    Phi1_bar = Phi1 .* Gmat + diag(kk_true(:));
    Phi2_bar = Phi2 .* Gmat;
    Phi_bar  = Phi1_bar \ Phi2_bar;
    [~,EigenValues] = eig(Phi_bar,'vector');
    theta_bar(nn,:) = sort((angle(EigenValues)/delta));
    
    ESPRIT_Theta = zeros(2,CNT);
    GESPRIT_Theta = zeros(2,CNT);

    for cnt = 1:1:CNT
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
        end
        gg = (1-c*ell_estim.^(-2))./(1+c*ell_estim.^(-1));  

        ESPRIT_Theta(:,cnt) = GetESPRITE(U_S,n,delta);
        GESPRIT_Theta(:,cnt) = GetGESPRITE(U_S,n,delta,gg,k);
    end

    theta_hat(nn,1)=mean(ESPRIT_Theta(1,:));
    theta_hat(nn,2)=mean(ESPRIT_Theta(2,:));
    theta_tilde(nn,1) = mean(GESPRIT_Theta(1,:));
    theta_tilde(nn,2)=mean(GESPRIT_Theta(2,:));
end

figure()
hold on;
plot(N_list,theta_hat(:,1).');
plot(N_list,theta_hat(:,2).');
plot(N_list,theta_tilde(:,1).');
plot(N_list,theta_tilde(:,2).');
plot(N_list,theta_cc(:,1).');
plot(N_list,theta_cc(:,2).');
plot(N_list,theta_bar(:,1).');
plot(N_list,theta_bar(:,2).');
legend('ESPRIT-1','ESPRIT-2','GESPRIT-1','GESPRIT-2','True-1','True-2','bar1','bar2');
