function [ Result ] = evalt(Fpred, Ygnd, thr, flag)
%%
% Fpred: L*N predicted values
% Ypred: L*N predicted labels
% Ygnd: L*N groundtruth labels
% thr: threshold value
% flag: default value is true
%%
if flag
    % default
    Ypred = sign(Fpred);
else
    Ypred = sign(Fpred-thr);
end

%%
% Average Precision
Result.AveragePrecision = Average_precision(Fpred,Ygnd);

% Coverage
Result.Coverage = coverage(Fpred,Ygnd);

% One Error
Result.OneError = One_error(Fpred,Ygnd);

% Ranking Loss 
Result.RankingLoss = Ranking_loss(Fpred,Ygnd);

% Hamming Loss
Result.HammingLoss = Hamming_loss(Ypred,Ygnd);

% Average AUC
Result.AvgAuc = avgauc(Fpred,Ygnd);