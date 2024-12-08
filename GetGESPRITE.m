function Res= GetGESPRITE(U_S,n,delta,gg,kk)
    [N,~] = size(U_S);
    J_tmp = eye(N);
    J1 = J_tmp(1:n,:);
    J2 = J_tmp(1+delta:n+delta,:);
    Phi1 =  U_S' * J1' * J1 * U_S;
    Phi2 = U_S' * J1' * J2 * U_S;
    g1 = gg(1,:);  k1 = kk(1,:);
    g2 = gg(2,:);  k2 = kk(2,:);
    Tilde_Phi1 = Phi1.*[1/g1,1/sqrt(g1*g2);1/sqrt(g1*g2),1/g2]-diag([k1/g1,k2/g2]);
    Tilde_Phi2 = Phi2.*[1/g1,1/sqrt(g1*g2);1/sqrt(g1*g2),1/g2];
    Tilde_Phi = Tilde_Phi1\Tilde_Phi2;
    [~,EigenValues] = eig(Tilde_Phi,'vector');
    Res = sort((angle(EigenValues)/delta));
end

