function l = log_gaussian_pdf(X, mu, Sigma)
% Multivariate Gaussian log-pdf
[N,d] = size(X);
mu = mu(:)'; 
[U,p] = chol(Sigma);
if p>0, error('Sigma must be positive definite'); end
logdetS = 2*sum(log(diag(U)));
Xc = X - mu;
invS = inv(Sigma);
quad = sum((Xc*invS).*Xc, 2);
l = -0.5*(d*log(2*pi) + logdetS + quad);
end