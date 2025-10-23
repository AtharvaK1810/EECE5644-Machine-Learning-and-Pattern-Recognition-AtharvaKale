function out = q2_a2_main(name,nuid,email)

rng(7,'twister');
Ntr=800; Nva=400;

phi = @(X)[ones(size(X,1),1), X, X(:,1).^2, X(:,1).*X(:,2), X(:,2).^2, ...
           X(:,1).^3, X(:,1).^2.*X(:,2), X(:,1).*X(:,2).^2, X(:,2).^3];
genX=@(N)randn(N,2);
w_true=[1; 0.6; -0.3; 0.4; -0.2; 0.3; 0.1; -0.05; 0.02; -0.08];
sigma=0.2;

Xtr=genX(Ntr); Ztr=phi(Xtr); ytr=Ztr*w_true + sigma*randn(Ntr,1);
Xva=genX(Nva); Zva=phi(Xva); yva=Zva*w_true + sigma*randn(Nva,1);

% ML
w_ml=(Ztr'*Ztr)\(Ztr'*ytr);
mse_ml=mean((Zva*w_ml - yva).^2);

% MAP with N(0,gamma I) prior => ridge with lambda = sigma^2/gamma
gammas = logspace(-6,2,25);
mse_map = zeros(numel(gammas),1);
for i=1:numel(gammas)
    lam = (sigma^2)/gammas(i);
    w_map=(Ztr'*Ztr + lam*eye(size(Ztr,2)))\(Ztr'*ytr);
    mse_map(i)=mean((Zva*w_map - yva).^2);
end
[best_mse,ib]=min(mse_map);

if ~exist('figs_A2','dir'), mkdir figs_A2; end
figure('Color','w');
semilogx(gammas, mse_map, '-o','LineWidth',1.4); hold on;
yline(mse_ml,'--','ML MSE');
xlabel('\gamma'); ylabel('Validation MSE'); title('A2–Q2 — MAP vs \gamma'); grid on;
add_signature(name,nuid,email);
saveas(gcf,'figs_A2/Q2_MAP_MSE_vs_gamma.pdf');

if ~exist('results','dir'), mkdir results; end
writematrix(mse_ml,'results/Q2_ML_MSE.csv');
writematrix([gammas(:) mse_map(:)],'results/Q2_MAP_gamma_MSE.csv');

out.mse_ml=mse_ml; out.best_mse_map=best_mse; out.best_gamma=gammas(ib);
end