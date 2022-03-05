%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%The MATLAB code of the paper "Weight matrix sharing for multi-label learning"
%version 1.0
%Contact: kqian2020@163.com or minfan@swpu.edu.cn
%Data set: 1. Mulan: http://mulan.sourceforge.net/datasets-mlc.html
%          2. Yahoo (LAMDA): http://www.lamda.nju.edu.cn/files/MDDM-expdata.rar
%          3. PALM (Image): http://palm.seu.edu.cn/zhangml/files/Image.rar
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;clc;
addpath(genpath('.'));
%load data
load('Art.mat');
%% parameter
parameter.lambda           = 10^-3;
parameter.lambda2          = 10^-3;
parameter.lambda3          = 10^-4;
parameter.alpha            = 0.6;
parameter.minLoss          = 10^-4;
parameter.rho              = 1.1;
parameter.mu               = 10^-6;
parameter.maxMu            = 10^6;
parameter.epsilon          = 10^-6;
parameter.maxIter          = 70;

%% perpare data
data    = [train_data;test_data];
target  = double([train_target,test_target]);
[DN,~] = size(data);
[~,TN] = size(target);
data = [data, ones(DN,1)];

%% cross validation
assert(DN==TN, 'Dimensional inconsistency')
runTimes = 5;
cross_num = 5;
All_results = zeros(6, cross_num*runTimes);
for r = 1:runTimes
    A = (1:DN)';
    indices = crossvalind('Kfold', A(1:DN,1), cross_num);
    for k = 1:cross_num 
        test = (indices == k);
        test_ID = find(test==1);
        train_ID = find(test==0);
        
        TE_data = data(test_ID,:);
        TR_data = data(train_ID,:);
        TE_target = target(:,test_ID);
        TR_target = target(:,train_ID);
        
        %get missing labels
        [J] = genObv(TR_target, 1);
        
        % train
        modelTrain  = train(J', TR_data, TR_target', parameter);
        
        %prediction and evaluation
        zz = mean(TE_target);
        TE_target(:,zz==-1) = [];
        TE_data(zz==-1,:) = [];
        [Output,results] = Predict(modelTrain, TE_data, TE_target);
        
        All_results(1,(r-1)*cross_num+k) = results.AveragePrecision;
        All_results(2,(r-1)*cross_num+k) = results.AvgAuc;
        All_results(3,(r-1)*cross_num+k) = results.HammingLoss;
        All_results(4,(r-1)*cross_num+k) = results.Coverage;
        All_results(5,(r-1)*cross_num+k) = results.OneError;
        All_results(6,(r-1)*cross_num+k) = results.RankingLoss;
        
    end
end
average_std = [mean(All_results,2) std(All_results,1,2)];
PrintResults(average_std);