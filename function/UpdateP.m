function [P] = UpdateP(J, X, Y)
L = X*X';
[N, La] = size(X);
tmpL = L - repmat(mean(L,1),N,1);
HLH = tmpL - repmat(mean(tmpL,2),1,N);
S = X' * HLH * X;
B = eye(La);
[tmp_P,~] = eig(S,B);
P = tmp_P;
end