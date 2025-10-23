function out = q1_a2_main(name,nuid,email)

p0 = 0.6; p1 = 0.4;
m01 = [-0.9; -1.1];  m02 = [ 0.8;  0.75];
m11 = [-1.1;  0.9];  m12 = [ 0.9; -0.75];
C   = [0.75 0; 0 1.25];

rng(42,'twister');
sample_mix = @(N,mA,mB) (rand(N,1)<0.5).*mvnrnd(mA',C,N) + (rand(N,1)>=0.5).*mvnrnd(mB',C,N);
gen_data = @(N) deal( ...
    [ sample_mix(round(N*p0),m01,m02);  sample_mix(round(N*p1),m11,m12) ], ...
    [ zeros(round(N*p0),1);               ones(round(N*p1),1) ]);

[X50 ,y50 ] = gen_data(50);
[X500,y500] = gen_data(500);
[X5k ,y5k ] = gen_data(5000);
[Xv  ,yv  ] = gen_data(10000);   % validation

if ~exist('results','dir'), mkdir results; end
if ~exist('figs_A2','dir'), mkdir figs_A2; end

% ---- MAP (true mixture pdfs) ----
lp0 = log(p0) + log(0.5) + logsumexp([log_gaussian_pdf(Xv,m01',C), log_gaussian_pdf(Xv,m02',C)],2);
lp1 = log(p1) + log(0.5) + logsumexp([log_gaussian_pdf(Xv,m11',C), log_gaussian_pdf(Xv,m12',C)],2);
score = lp1 - lp0;
pos = score(yv==1); neg=score(yv==0);
ths = linspace(min([pos;neg]), max([pos;neg]), 2000)';   % explicit thresholds
[TPR,FPR] = roc_from_scores(pos,neg,ths);
perr = p0*FPR + p1*(1-TPR);
[perr_min, idx] = min(perr);

figure('Color','w');
plot(FPR,TPR,'LineWidth',2); hold on;
plot(FPR(idx),TPR(idx),'ro','MarkerFaceColor','r','MarkerSize',5);
xlabel('FPR'); ylabel('TPR'); title('A2–Q1 — MAP ROC (validation)'); grid on;
add_signature(name,nuid,email);
saveas(gcf,'figs_A2/Q1_MAP_ROC.pdf');

% ---- Logistic (Linear) ----
phi_lin = @(X)[ones(size(X,1),1) X];
sigm    = @(z) 1./(1+exp(-z));
nll     = @(w,PHI,y) -sum(y.*log(sigm(PHI*w)) + (1-y).*log(1-sigm(PHI*w)));
train   = @(X,y,w0,phi) fminsearch(@(w)nll(w,phi(X),y), w0, optimset('Display','off'));
w50  = train(X50 ,y50 ,zeros(3,1),@(X)phi_lin(X));
w500 = train(X500,y500,zeros(3,1),@(X)phi_lin(X));
w5k  = train(X5k ,y5k ,zeros(3,1),@(X)phi_lin(X));
pe_lin = zeros(3,1);
for i=1:3
    w = {w50,w500,w5k}; w = w{i};
    p1hat = sigm(phi_lin(Xv)*w); yhat = p1hat>=0.5;
    pe_lin(i) = mean(yhat~=yv);
end

% ---- Logistic (Quadratic) ----
phi_quad = @(X)[ones(size(X,1),1), X(:,1),X(:,2), X(:,1).^2, X(:,1).*X(:,2), X(:,2).^2];
w50q  = train(X50 ,y50 ,zeros(6,1),@(X)phi_quad(X));
w500q = train(X500,y500,zeros(6,1),@(X)phi_quad(X));
w5kq  = train(X5k ,y5k ,zeros(6,1),@(X)phi_quad(X));
pe_quad = zeros(3,1);
for i=1:3
    w = {w50q,w500q,w5kq}; w = w{i};
    p1hat = sigm(phi_quad(Xv)*w); yhat = p1hat>=0.5;
    pe_quad(i) = mean(yhat~=yv);
end

% ---- Save ----
writematrix(perr_min,'results/Q1_MAP_minPe.csv');
writematrix([pe_lin pe_quad],'results/Q1_logistic_errors.csv');

out.pe_map  = perr_min;
out.pe_lin  = pe_lin;
out.pe_quad = pe_quad;
end