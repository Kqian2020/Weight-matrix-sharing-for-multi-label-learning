function [J] = genObv( train_target, rho )
% num_label*num_instance
% rho: 1 - missing rate
%GENERATEOBV Summary of this function goes here
%   Detailed explanation goes here
     if rho == 1
         J = ones(size(train_target));
         return;
     end
     J = zeros(size(train_target));
     for i=1:size(train_target,1)
         y = train_target(i,:);
         pos = find(y==1);
         neg = find(y~=1);
         pi = randperm(length(pos));
         pi = pi(1:ceil(rho*length(pos)));
         J(i,pos(pi)) = 1;
         ni = randperm(length(neg));
         ni = ni(1:ceil(rho*length(neg)));
         J(i,neg(ni)) = 1;
     end
end

%% References
% @article{Zhu2018Glocal,
% 	title		=		{Multi-Label Learning with Global and Local Label Correlation},
% 	author		=		{Yue Zhu and James T. Kwok and Zhi-Hua Zhou},
% 	journal		=		{IEEE Transactions on Knowledge and Data Engineering},
% 	year		=		{2018},
% 	volume		=		{30},
% 	number		=		{6},
% 	pages		=		{1081--1094},
% 	doi			=		{10.1109/TKDE.2017.2785795},
% 	url			=		{https://doi.org/10.1109/TKDE.2017.2785795}
% }