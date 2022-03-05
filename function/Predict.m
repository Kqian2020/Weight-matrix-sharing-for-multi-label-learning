function [TY,results] = Predict(modelTrain, Xt, Yt)
TY = Xt*modelTrain.W;
results =  evalt(TY', Yt, (max(TY(:))-min(TY(:)))/2, 1);
end    