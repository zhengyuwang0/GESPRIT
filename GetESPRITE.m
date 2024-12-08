function Res= GetESPRITE(U_S,n,delta)
    [N,~] = size(U_S);
    J_tmp = eye(N);
    J1 = J_tmp(1:n,:);
    J2 = J_tmp(1+delta:n+delta,:);
    Phi =  (U_S' * J1' * J1 * U_S) \(U_S' * J1' * J2 * U_S);
    [~,EigenValues] = eig(Phi,'vector');
    Res = sort((angle(EigenValues)/delta));
end

