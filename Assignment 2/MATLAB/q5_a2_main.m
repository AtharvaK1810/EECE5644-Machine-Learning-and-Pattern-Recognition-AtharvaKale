function q5_a2_main(name,nuid,email)

rng(5);
K=4; N=100;
theta_true=[0.10 0.20 0.50 0.20];

% sample labels 1..K
Z=randsample(1:K,N,true,theta_true);
Nk=histcounts(Z,1:K+1);

% ML
theta_ML = Nk/sum(Nk);

% MAP with a few priors
alphas = { [1 1 1 1], [2 2 2 2], [5 5 5 5] };
theta_MAP = zeros(numel(alphas), K);
for i=1:numel(alphas)
    a = alphas{i};
    theta_MAP(i,:) = (Nk + a - 1) / (N + sum(a) - K);
end

if ~exist('figs_A2','dir'), mkdir figs_A2; end
figure('Color','w');
bar([theta_true; theta_ML; theta_MAP], 'grouped');
legend({'True','ML','MAP(\alpha=1)','MAP(\alpha=2)','MAP(\alpha=5)'}, 'Location','northoutside');
xlabel('Category k'); ylabel('\theta_k'); title('A2–Q5: Categorical–Dirichlet Estimates');
grid on; add_signature(name,nuid,email);
saveas(gcf,'figs_A2/Q5_Categorical_Dirichlet.pdf');

if ~exist('results','dir'), mkdir results; end
T = [theta_true; theta_ML; theta_MAP];
writematrix(T,'results/Q5_theta_estimates.csv');
end