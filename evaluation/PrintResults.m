
function PrintResults(Result)
fprintf('------------------------------------------------\n');

fprintf('Evalucation Metric          Mean     Std\n');
fprintf('-------------------------------------------------\n');
fprintf('AveragePrecision           %.4f  %.4f\r',Result(1,1),Result(1,2));
fprintf('AvgAuc                     %.4f  %.4f\r',Result(2,1),Result(2,2));
fprintf('HammingLoss                %.4f  %.4f\r',Result(3,1),Result(3,2));
fprintf('Coverage                   %.4f  %.4f\r',Result(4,1),Result(4,2));
fprintf('OneError                   %.4f  %.4f\r',Result(5,1),Result(5,2));
fprintf('RankingLoss                %.4f  %.4f\r',Result(6,1),Result(6,2));

fprintf('--------------------------------------------------\n');
end