%% right
clear; clc
sigma2 = 1;
nb_Loop = 200;
% P = [2, 0.8;0.8,2];
P = 2*[1,0;0,1];
% theta_true = [0,pi/4];

coeff = 1:1:5;
ESPRIT_MSE = zeros(nb_Loop,length(coeff));
GESPRIT_MSE = zeros(nb_Loop,length(coeff));
CRB_Theory = zeros(1,length(coeff));
ESPRIT_estimate = zeros(2,nb_Loop);
GESPRIT_estimate = zeros(2,nb_Loop);

for it = 1:1:length(coeff)
    N = 50*2^(coeff(it));
    n = floor(2*N/3);
    delta = floor(N/3);
    % delta = 1;
    % n = N-1;
    
    T  = 2*N;
    c = N/T;
    theta_true = [0,0.8*2*pi/N];
%     theta_true = [0,pi/4];
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

        ESPRIT_Theta = GetESPRITE(U_S,n,delta);   % 2*1
        GESPRIT_Theta = GetGESPRITE(U_S,n,delta,gg,kk);  % 2*1
        
        ESPRIT_MSE(jj,it) = sum((ESPRIT_Theta.' - theta_true).^2) / k;
        ESPRIT_estimate(:,jj) = ESPRIT_Theta;

        GESPRIT_MSE(jj,it) = sum((GESPRIT_Theta.' - theta_true).^2) / k;
        GESPRIT_estimate(:,jj) = GESPRIT_Theta;

    end
    ESPRIT_Bias_E(it) = sum((mean(ESPRIT_estimate,2)-theta_true.').^2)/2;
    GESPRIT_Bias_E(it) = sum((mean(GESPRIT_estimate,2)-theta_true.').^2)/2;
    CRB = sigma2 / (2*T) *inv(real(Df'*(eye(N)-A*inv(A'*A)*A')*Df) .*P);
    CRB_Theory(1,it) = trace(CRB)/2;
end

ESPRIT_MSE_E  = mean(ESPRIT_MSE,1);
ESPRIT_Var_E  = ESPRIT_MSE_E - ESPRIT_Bias_E;
GESPRIT_MSE_E  = mean(GESPRIT_MSE,1);
GESPRIT_Var_E  = GESPRIT_MSE_E - GESPRIT_Bias_E;


figure;
hold on;
plot(50*2.^(coeff),log10(GESPRIT_MSE_E));
plot(50*2.^(coeff),log10(GESPRIT_Var_E));
plot(50*2.^(coeff),log10(ESPRIT_MSE_E))
plot(50*2.^(coeff),log10(ESPRIT_Var_E))
plot(50*2.^(coeff),log10(CRB_Theory))
legend('GESPRIT-MSE','GESPRIT-VAR','ESPRIT-MSE','ESPRIT-VAR','CRB');
xlabel('N');
ylabel('MSE(dB)');

%% left
clear; clc;
N = 80;
T  = 160;
c = N/T;   % sqrt(0.25)=0.5
CNT = 500;
n = floor(2*N/3);
delta = floor(N/3);
% delta = 1;
% n = N-1;
theta_true = [0,0.8*2*pi/N];
% theta_true = [0,pi/4];
k = length(theta_true);
P = 2*[1,0;0,1];
% P = [2, 0.8;0.8,2];

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

gg = (1-c*rho.^(-2))./(1+c*rho.^(-1));   % 2*1
kk = n/N*(c+c*rho.^(-1))./(c+rho);

% theta_bar
J_tmp = eye(N);
J1 = J_tmp(1:n,:);
J2 = J_tmp(1+delta:n+delta,:);
Phi1 =  V' * (J1' * J1) * V;
Phi2 = V' * J1' * J2 * V;
bar_Phi1 = Phi1.*[gg(1,:),sqrt(gg(1,:)*gg(2,:));sqrt(gg(1,:)*gg(2,:)),gg(2,:)]+diag([kk(1,:),kk(2,:)]);
bar_Phi2 = Phi2.*[gg(1,:),sqrt(gg(1,:)*gg(2,:));sqrt(gg(1,:)*gg(2,:)),gg(2,:)];
Tilde_Phi = inv(bar_Phi1)*bar_Phi2;
[~,EigenValues] = eig(Tilde_Phi,'vector');
theta_bar = sort((angle(EigenValues)/delta));

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
    gg_1 = (1-c*ell_estim.^(-2))./(1+c*ell_estim.^(-1));  
    kk_1 = n/N*(c+c*ell_estim.^(-1))./(c+ell_estim);
    
    ESPRIT_Theta(:,cnt) = GetESPRITE(U_S,n,delta);
    GESPRIT_Theta(:,cnt) = GetGESPRITE(U_S,n,delta,gg_1,kk_1);
end

theta_hat(:,1)=mean(ESPRIT_Theta(1,:));
theta_hat(:,2)=mean(ESPRIT_Theta(2,:));
theta_tilde(:,1) = mean(GESPRIT_Theta(1,:));
theta_tilde(:,2)=mean(GESPRIT_Theta(2,:));


mmin = min([mean(ESPRIT_Theta(1,:)) mean(ESPRIT_Theta(2,:)) mean(GESPRIT_Theta(1,:)) mean(GESPRIT_Theta(2,:))]);
mmax = max([mean(ESPRIT_Theta(1,:)) mean(ESPRIT_Theta(2,:)) mean(GESPRIT_Theta(1,:)) mean(GESPRIT_Theta(2,:))]);
figure()
x = mmin-0.001:0.0001:mmax+0.001;
hold on
xline(theta_true(1),'LineWidth',2,'LineStyle','--');  
xline(theta_true(2),'LineWidth',2,'LineStyle','--');  
xline(theta_hat(:,1),'LineWidth',2,'LineStyle','-','color','g');  
xline(theta_hat(:,2),'LineWidth',2,'LineStyle','-','color','g'); 
xline(theta_tilde(:,1),'LineWidth',2,'LineStyle','-','color','r');  
xline(theta_tilde(:,2),'LineWidth',2,'LineStyle','-','color','r'); 
legend('theta true','theta true','ESPRIT','ESPRIT','GESPRIT','GESPRIT');
ylim([0,2])
xlim([mmin-0.01,mmax+0.01])
xlabel('theta(rad)')
hold off;
