function s = logsumexp(a, dim)

if nargin < 2
    dim = 1;
end

amax = max(a, [], dim);
s = amax + log(sum(exp(a - amax), dim));
% handle cases where all elements are -Inf
s(isinf(amax)) = -Inf;
end