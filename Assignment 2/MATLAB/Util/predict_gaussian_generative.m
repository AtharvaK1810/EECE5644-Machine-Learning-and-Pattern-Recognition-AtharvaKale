function [yhat, logpost] = predict_gaussian_generative(model, X)
means = model.means; covs = model.covs;
priors = model.priors; labels = model.labels;
K = numel(labels); N = size(X,1);
logpost = zeros(N,K);
for k=1:K
    logpost(:,k) = log(priors(k)) + log_gaussian_pdf(X, means{k}, covs{k});
end
[~, idx] = max(logpost, [], 2);
yhat = labels(idx)';
end