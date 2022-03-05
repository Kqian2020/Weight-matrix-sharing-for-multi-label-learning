function model = train( J, X, Y, parameter)
%% parameters
lambda     = parameter.lambda;
lambda2    = parameter.lambda2;
lambda3    = parameter.lambda3;
alpha      = parameter.alpha;
rho        = parameter.rho;
mu         = parameter.mu;
maxMu      = parameter.maxMu;
epsilon    = parameter.epsilon;
maxIter    = parameter.maxIter;

%% initialization
[num_instance,num_dim]  = size(X);
[~,num_class]  = size(Y);
Ymis = J.*Y;
W = eye(num_dim, num_class);
B = W; % Q
Lambda = W - B;
W_1 = W;

% label correlation C
C = pdist2(Ymis', Ymis', 'cosine');
L = diag(sum(C)) - C;

% feature correlation P
[Wd] = UpdateP(J, X, Y);
Ld = Wd*Wd'; % M

% representative
paraDc = 0.15;
[~, tempIndex] = InstanceRepresentativeness(X, paraDc);
YSub = Ymis(tempIndex(1:round(num_instance*0.2)),:);
YTY = YSub'*YSub;
XTX = X'*X;

Lip1 = 2*norm(XTX)^2;
Lip1 = Lip1 + 2*norm(2*lambda2*XTX)^2*norm(L)^2;
Lip1 = Lip1 + 2*norm(2*lambda3*Ld)^2*norm(YTY)^2;

iter = 1;
bk = 1;
bk_1 = 1;

while iter <= maxIter
    %% Lip
    Lip2 = Lip1 + 2*mu^2;
    Lip = sqrt(Lip2);
    
    %% update W
    W_k  = W + (bk_1 - 1)/bk * (W - W_1);
    Gw_x_k = W_k - 1/Lip * gradientOfW(J,XTX,YTY,W,L,Ld,B,Lambda,lambda2,lambda3,mu,X,Y);
    W_1 = W;
    W = softthres(Gw_x_k, (lambda*(1-alpha))/Lip);
    
    %% update B
    [U, Sigma, V] = svd(W + Lambda/mu,'econ');
    B = U * softthres(Sigma,(lambda*alpha)/mu) * V';
    
    %% update Lambda
    % Lmabda
    Lambda = Lambda + mu*(W - B);
    % mu
    mu = min(maxMu, mu*rho);
    % b
    bk_1 = bk;
    bk = (1 + sqrt(4*bk^2 + 1))/2;
    
    %% stop conditions
    if norm(W - B, 'inf') < epsilon
        break;
    end
    iter=iter+1;
end

model.W = W;
model.B = B;
model.iter = iter;
end

%% soft thresholding operator
function Ws = softthres(W,lambda)
Ws = max(W-lambda,0) - max(-W-lambda,0);
end
%% gradient W
function gradient = gradientOfW(J,XTX,YTY,W,L,Ld,B,Lambda,lambda2,lambda3,mu,X,Y)
gradient = X'*(J.*(X*W - Y));
gradient = gradient + 2*lambda2*XTX*W*L + 2*lambda3*Ld*W*YTY;
gradient = gradient + mu * (W - B) + Lambda;
end