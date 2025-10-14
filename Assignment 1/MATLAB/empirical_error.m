function pe = empirical_error(PF, PM, p0, p1)
% Empirical average error
pe = PF.*p0 + PM.*p1;
end