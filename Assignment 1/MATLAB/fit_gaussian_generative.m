function model = fit_gaussian_generative(X, y, lam)
if nargin<3, lam = 0; end
labels = unique(y(:)'); K = numel(labels);
[N,d] = size(X);
means = cell(1,K); covs = cell(1,K); priors = zeros(1,K);
for k=1:K
    lab = labels(k);
    Xi = X(y==lab, :);
    means{k} = mean(Xi,1);
    Ci = cov(Xi);
    if lam>0, Ci = Ci + lam*eye(d); end
    covs{k} = Ci;
    priors(k) = size(Xi,1)/N;
end
model.means = means; model.covs = covs;
model.priors = priors; model.labels = labels;
end