clear; clc;
N = 80;
T  = 160;

theta_true = [0,2*0.8*pi/N];
k = length(theta_true);
P = 2*eye(2,2);

sigma2 = 1;
clear i
a = @(theta) exp(1i*theta*(0:N-1)')/sqrt(N);
diffa = @(theta) exp(1i*theta*(0:N-1)') *1i .*(0:N-1).' /sqrt(N);
A = [];
for tmp_index=1:length(theta_true)
    A = [A a(theta_true(tmp_index))];
end
[V,E] = eig(A*P*A','vector');
[E, index1] = sort(E,'descend');
V = V(:, index1);
V = V(:,1:2);
rho = E(1:k)./sigma2;  % snr

ds = 1:1:round(pi/theta_true(2));
nb_Loop = 1000;
ESPRIT_MSE = zeros(nb_Loop,length(ds));
for it = 1:1:length(ds)
   delta = ds(it);
   n = N-delta;
   for jj = 1 : nb_Loop
        S = sqrt(1/2) *sqrtm(P)*(randn(k,T) + 1i *randn(k,T));
        Z = sqrt(sigma2/2) * (randn(N,T) + 1i* randn(N,T));
        X = A*S + Z;    
        SCM = X*(X')/T;
        [U,eigs_SCM] = eig(SCM,'vector');
        [eigs_SCM, index] = sort(eigs_SCM,'descend');
        U = U(:,index);
        U_S = U(:,1:k);
        ESPRIT_Theta = GetESPRITE(U_S,n,delta);  
        ESPRIT_MSE(jj,it) =(ESPRIT_Theta(2) - theta_true(2)).^2;
   end   
end
ESPRIT_MSE_E  = mean(ESPRIT_MSE,1);

figure()
hold on
plot(N-ds,log10(ESPRIT_MSE_E))
xline(2*N/3)
hold off
title('MSE with the length of subarray, N=80')