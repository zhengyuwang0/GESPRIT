clear; clc;
N = 400;
T  = 800;
c = N/T;   % sqrt(0.25)=0.5
nb_Loop = 50;
% n = N-1;
% delta = 1;
n = floor(2*N/3);
delta = floor(N/3);
theta_true = [0,0.8*2*pi/N];
% theta_true = [0,pi/4];
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
P1 = 2*[1,0;0,1];
[V,E] = eig(A*P1*A','vector');
[E, index1] = sort(E,'descend');
V = V(:, index1);
V = V(:,1:2);

SNRList = -4:2:14;
powList = db2pow(SNRList)/E(2);  
sigma2 = 1;
SNR_cond = pow2db(sqrt(c)/E(2));

ESPRIT_MSE = zeros(nb_Loop,length(SNRList));
MUSIC_MSE = zeros(nb_Loop,length(SNRList));
GMUSIC_MSE = zeros(nb_Loop,length(SNRList));
GESPRIT_MSE = zeros(nb_Loop,length(SNRList));

CRB_Theory = zeros(1,length(SNRList));

for it=1:1:length(SNRList)
    pow = powList(it);
    P = pow*P1;

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
            kk = n/N*(c+c*ell_estim.^(-1))./(c+ell_estim);
        else
            gg = ones(2,1); kk = zeros(2,1);
        end

        MUSIC_Theta = GetMusic(U_S);
        GMUSIC_Theta = GetGMusic(U_S,eigs_SCM,c);
        ESPRIT_Theta = GetESPRITE(U_S,n,delta);   % 2*1
        MLE_Theta = GetMLE(T,N,X);   % 2*1
        GESPRIT_Theta = GetGESPRITE(U_S,n,delta,gg,kk);   % 2*1
        MUSIC_MSE(jj,it) = (MUSIC_Theta(2) - theta_true(2)).^2;
        GMUSIC_MSE(jj,it) = (GMUSIC_Theta(2) - theta_true(2)).^2;
        ESPRIT_MSE(jj,it) =(ESPRIT_Theta(2) - theta_true(2)).^2;
        GESPRIT_MSE(jj,it) =(GESPRIT_Theta(2) - theta_true(2)).^2;

    end
    ESPRIT_MSE_E  = mean(ESPRIT_MSE,1);
    MUSIC_MSE_E  = mean(MUSIC_MSE,1);
    GMUSIC_MSE_E  = mean(GMUSIC_MSE,1);
    GESPRIT_MSE_E  = mean(GESPRIT_MSE,1);

    CRB = sigma2 / (2*T) *inv(real(Df'*(eye(N)-A*inv(A'*A)*A')*Df) .*P);
    % CRB_Theory(1,it) = trace(CRB)/k;
    CRB_Theory(1,it) = CRB(2,2);
end

figure();
hold on;
plot(SNRList,log10(MUSIC_MSE_E),'LineStyle','-','Color','#0072BD','Marker','.','LineWidth',1.5);
plot(SNRList,log10(GMUSIC_MSE_E),'LineStyle','-','Color','#77AC30','Marker','o','LineWidth',1.5);
plot(SNRList,log10(ESPRIT_MSE_E),'LineStyle','-','Color','#FF6100','Marker','*','LineWidth',1.5);
plot(SNRList,log10(GESPRIT_MSE_E),'LineStyle','-','Color','#A020F0','Marker','^','LineWidth',1.5);
plot(SNRList,log10(CRB_Theory),'LineStyle','--','Color','#77AC30','Marker','o','LineWidth',1.5);
xline(-2,'LineWidth',1.5,'LineStyle','--'); 
hold off;
legend('MUSIC MSE','GMUSIC MSE','ESPRIT MSE','GESPRIT MSE','CRB')
xlabel('N')
ylabel('log10')





