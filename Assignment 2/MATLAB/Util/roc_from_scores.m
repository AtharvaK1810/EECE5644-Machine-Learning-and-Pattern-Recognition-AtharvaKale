function [TPR,FPR] = roc_from_scores(pos_scores, neg_scores, thresholds)
TPR = zeros(numel(thresholds),1);
FPR = zeros(numel(thresholds),1);
for i = 1:numel(thresholds)
    t = thresholds(i);
    TPR(i) = mean(pos_scores >= t);
    FPR(i) = mean(neg_scores >= t);
end
end