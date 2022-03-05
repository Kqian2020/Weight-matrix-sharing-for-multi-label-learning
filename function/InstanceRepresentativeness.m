function [representativenessArray, representativenessRank] = InstanceRepresentativeness(X, paraDc)
[num_instance, ~] = size(X);
tempDistanceMatrix = pdist2(X, X,'euclidean');
tempDensityArray = sum(exp(-tempDistanceMatrix.^2./ paraDc^2));

tempDistanceToMasterArray = zeros(1, num_instance);
for i = 1:num_instance
    tempIndex = tempDensityArray>tempDensityArray(i);
    if sum(tempIndex) > 0
        tempDistanceToMasterArray(1,i) = min(tempDistanceMatrix(i,tempIndex)); 
    end
end

representativenessArray = tempDensityArray.*tempDistanceToMasterArray;
[~, representativenessRank] = sort(representativenessArray, 'descend');
end