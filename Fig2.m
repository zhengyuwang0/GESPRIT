close all; clear; clc
sigma2 = 1;
CNT = 100;
coeff = 1:2:17;
err = zeros(CNT,length(coeff));
for it = 1:1:length(coeff)
    N = 128*coeff(it);
    T  = 2*N;
    c = N/T;  
    
    % closely-doa case
    theta_true = [0,0.8*2*pi/N];
    k = length(theta_true);
    n = floor(N*2/3);
    delta = floor(N/3);
    P = 2*[1,0;0,1];
    
    % widely-doa case
%     theta_true = [0,pi/4]; 
%     P = [2, 0.8;0.8,2];
%     k = length(theta_true);
%     n = N-1;
%     delta = 1;
    
    J = eye(N);
    J1 = J(1:n,:);
    J2 = J(1+delta:n+delta,:);   % delta = 1

    clear i
    a = @(theta) exp(1i*theta*(0:N-1)')/sqrt(N);
    A = [];
    for tmp_index=1:length(theta_true)
        A = [A a(theta_true(tmp_index))];
    end
    [V,D] = eig(A*P*A','vector');
    [D, index1] = sort(D,'descend');
    V = V(:, index1);
    V = V(:,1:k);
    rho = D(1:k)/sigma2;
%     gg_true = (1-c*rho.^(-2))./(1+c*rho.^(-1));   % 2*1
%     kk_true = n/N*(c+c*rho.^(-1))./(c+rho);
    
    Phi1_true = V'*(J1'*J1)*V;
    Phi2_true = V'*J1'*J2*V;
    Phi_true = Phi1_true\Phi2_true;
    
    for cnt = 1:1:CNT
        S = sqrtm(P)*randn(k,T);    % K*T 
        Z = complex(randn(N,T), randn(N,T));
        X = A*S + sqrt(sigma2/2)*Z;     % N*T
        SCM = X*(X')/T;
        [U,eigs_SCM] = eig(SCM,'vector');
        [eigs_SCM, index] = sort(eigs_SCM,'descend');
        U = U(:, index);
        U_S = U(:,1:k);
        
        Phi1_hat = U_S'*(J1'*J1)*U_S;
        Phi2_hat = U_S'*J1'*J2*U_S; 
        Phi_hat = Phi1_hat\Phi2_hat;   % Phi1\Phi2  

        lambda_bar = eigs_SCM(1:k);
        if lambda_bar>=(1+sqrt(c))^2
            ell_estim = (lambda_bar-(1+c))/2 + sqrt((lambda_bar-(1+c)).^2 - 4*c)/2;
        end
        gg = (1-c*ell_estim.^(-2))./(1+c*ell_estim.^(-1));  
        kk = n/N*(c+c*ell_estim.^(-1))./(c+ell_estim);
        g1 = gg(1,:);  k1 = kk(1,:);
        g2 = gg(2,:);  k2 = kk(2,:);

%% test sp-norm between Phi_hat and Phi_bar
        % Phi1_bar = Phi1_true.*[g1,sqrt(g1*g2);sqrt(g1*g2),g2]+diag([k1,k2]);
        % Phi2_bar = Phi2_true.*[g1,sqrt(g1*g2);sqrt(g1*g2),g2];
        % Phi_bar = Phi1_bar\Phi2_bar;
        % eig1 = eig(Phi_hat); eig2 = eig(Phi_bar);
        % [va1,ind1] = sort(real(eig1));[va2,ind2] = sort(real(eig2));
        % eig1 = eig1(ind1,:); eig2 = eig2(ind2,:);

%% test sp-norm between Phi_hatG and Phi_true
        Tilde_Phi1 = Phi1_hat.*[1/g1,1/sqrt(g1*g2);1/sqrt(g1*g2),1/g2]-diag([k1/g1,k2/g2]);
        Tilde_Phi2 = Phi2_hat.*[1/g1,1/sqrt(g1*g2);1/sqrt(g1*g2),1/g2];
        Tilde_Phi = Tilde_Phi1\Tilde_Phi2;
        eig1 = eig(Phi_true); eig2 = eig(Tilde_Phi);
        [va1,ind1] = sort(real(eig1));[va2,ind2] = sort(real(eig2));
        eig1 = eig1(ind1,:); eig2 = eig2(ind2,:);
        
        err(cnt,it) = abs(eig1(2)-eig2(2));
    end

end
sp_norm = [mean(err,1);std(err,0,1)];
