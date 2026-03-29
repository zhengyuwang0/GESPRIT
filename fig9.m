%% test coherence
clear; clc;
coeff_list = 1:1:12;
nb_Loop = 2000;
klist = [2,4,8];
N = 400;
T = 800;
c = N/T;
sigma2 = 1;
ESPRIT_MSE = zeros(nb_Loop,1);
MUSIC_MSE = zeros(nb_Loop,1);
GMUSIC_MSE = zeros(nb_Loop,1);
GESPRIT_MSE = zeros(nb_Loop,1);
CRB_Theory = zeros(3,length(coeff_list));
ESPRIT_MSE_E  = zeros(3,length(coeff_list));
MUSIC_MSE_E  = zeros(3,length(coeff_list));
GMUSIC_MSE_E  = zeros(3,length(coeff_list));
GESPRIT_MSE_E  = zeros(3,length(coeff_list));

for it = 1:length(coeff_list)
    dd = coeff_list(it)*0.1*2*pi/N;
    for ii = 1:1:length(klist)
        k = klist(ii);
        testcase = k; 
        switch testcase
            case 4
                theta_true = [0,dd,2*dd,3*dd];
                theta_true1 = todeg(theta_true);
            case 8
                theta_true = [0,dd,2*dd,3*dd,4*dd,5*dd,6*dd,7*dd];
                theta_true1 = todeg(theta_true);
            case 2
            theta_true = [0,dd];
            theta_true1 = todeg(theta_true);
        end 
        delta = 120;
        n = N-delta;
        rr = 0.2;
        P1 = (1-rr)*eye(k) + rr*ones(k);
        
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
        [V,D] = eig(A*P1*A','vector');
        [D, index1] = sort(D,'descend');
        V = V(:, index1);
        V = V(:,1:k);
        ell_target = 3; 
        alpha = ell_target * sigma2 / D(k);
        P = alpha * P1;

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
            eigs_SCM(k)
            if lambda_bar>=(1+sqrt(c))^2
                ell_estim = (lambda_bar-(1+c))/2 + sqrt((lambda_bar-(1+c)).^2 - 4*c)/2;
                gg = (1-c*ell_estim.^(-2))./(1+c*ell_estim.^(-1));  
                kk = 1-gg;
            else
                gg = ones(k,1); kk = zeros(k,1);
            end

            GMUSIC_Theta = GetGMusic(U_S,eigs_SCM,c);
            GMUSIC_Theta = todeg(GMUSIC_Theta);
            GMUSIC_MSE(jj) = sum((GMUSIC_Theta - theta_true1).^2)/k; 
            GESPRIT_Theta = GetGESPRITE(U_S,n,delta,gg,k);   % 2*1
            GESPRIT_Theta = todeg(GESPRIT_Theta);    
            GESPRIT_MSE(jj) = sum((GESPRIT_Theta.' - theta_true1).^2)/k;   
        end
        GMUSIC_MSE_E(ii,it) = mean(GMUSIC_MSE,1);
        GESPRIT_MSE_E(ii,it)  = mean(GESPRIT_MSE,1);
    
    end

end

figure();
hold on;
plot(coeff_list,log10(GMUSIC_MSE_E(1,:)),'LineStyle','-','Color','#77AC30','Marker','o','LineWidth',1.5);
plot(coeff_list,log10(GESPRIT_MSE_E(1,:)),'LineStyle','-','Color','#77AC30','Marker','x','LineWidth',1.5);
plot(coeff_list,log10(GMUSIC_MSE_E(2,:)),'LineStyle','-','Color','#A020F0','Marker','o','LineWidth',1.5);
plot(coeff_list,log10(GESPRIT_MSE_E(2,:)),'LineStyle','-','Color','#A020F0','Marker','x','LineWidth',1.5);
plot(coeff_list,log10(GMUSIC_MSE_E(3,:)),'LineStyle','-','Color','#0072BD','Marker','o','LineWidth',1.5);
plot(coeff_list,log10(GESPRIT_MSE_E(3,:)),'LineStyle','-','Color','#0072BD','Marker','x','LineWidth',1.5);
legend('GMUSIC-1','GESPRIT-1')
hold off;
xlabel('ds')
ylabel('log10')
