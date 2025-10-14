function q1_main()

addpath('/Users/atharva/Documents/MS in IoT/Sem 3/Machine Learning and Pattern Recognition/Assignment 1/Matlab');

rng(42,'twister');
N = 10000;
p0 = 0.65; p1 = 0.35;

m0 = [-0.5 -0.5 -0.5];
C0 = [1 -0.5 0.3; -0.5 1 -0.5; 0.3 -0.5 1];
m1 = [1 1 1];
C1 = [1 0.3 -0.2; 0.3 1 0.3; -0.2 0.3 1];

u  = rand(N,1) >= p0;              % 1 => class 1
N0 = sum(~u);  N1 = sum(u);
X0 = mvnrnd(m0, C0, N0);
X1 = mvnrnd(m1, C1, N1);
X  = [X0; X1];
y  = [zeros(N0,1); ones(N1,1)];
perm = randperm(N);
X = X(perm,:); y = y(perm);

if ~exist('figs_Q1','dir'), mkdir figs_Q1; end

% ===================== 1A: TRUE model ERM =====================
logp1 = log_gaussian_pdf(X, m1, C1);
logp0 = log_gaussian_pdf(X, m0, C0);
llr   = logp1 - logp0;

ths = linspace(min(llr)-5, max(llr)+5, 1200)';
pos = llr(y==1); neg = llr(y==0);
[TPR, FPR] = roc_from_scores(pos, neg, ths);
PD = TPR; PFA = FPR; PM = 1 - PD;
perrs = empirical_error(PFA, PM, p0, p1);
[perr_min, idx] = min(perrs);
thr_star        = ths(idx);
log_gamma_star  = log(p0/p1);

fprintf('Q1.A  min Pe = %.6f, FPR=%.6f, TPR=%.6f, thr_log=%.6f, theory_log_gamma=%.6f\n', ...
    perr_min, PFA(idx), PD(idx), thr_star, log_gamma_star);

figure; plot(FPR,TPR,'LineWidth',2); hold on;
plot(PFA(idx),PD(idx),'o','MarkerSize',8,'LineWidth',1.5);
xlabel('False Positive  P(D=1|L=0)'); ylabel('True Positive  P(D=1|L=1)');
title('Q1.A ROC — ERM (true pdf)'); grid on;

% --- add signature ---
add_signature();

saveas(gcf, fullfile('figs_Q1','Q1A_ROC_true.pdf'));

% ===================== 1B: Naive Bayes (identity Σ) =====================
w  = (m1 - m0)'; 
b  = -0.5*(m1*m1' - m0*m0');
score_nb = X*w + b;

ths_nb = linspace(min(score_nb)-5, max(score_nb)+5, 1200)';
pos_nb = score_nb(y==1); neg_nb = score_nb(y==0);
[TPR_nb, FPR_nb] = roc_from_scores(pos_nb, neg_nb, ths_nb);
PD_nb = TPR_nb; PFA_nb = FPR_nb; PM_nb = 1 - PD_nb;
perrs_nb = empirical_error(PFA_nb, PM_nb, p0, p1);
[perr_min_nb, idx_nb] = min(perrs_nb);
thr_star_nb = ths_nb(idx_nb);

fprintf('Q1.B  min Pe = %.6f, FPR=%.6f, TPR=%.6f, thr=%.6f\n', ...
    perr_min_nb, PFA_nb(idx_nb), PD_nb(idx_nb), thr_star_nb);

figure; plot(FPR_nb,TPR_nb,'LineWidth',2);
xlabel('False Positive  P(D=1|L=0)'); ylabel('True Positive  P(D=1|L=1)');
title('Q1.B ROC — Naive Bayes (I)'); grid on;

% --- add signature ---
add_signature();

saveas(gcf, fullfile('figs_Q1','Q1B_ROC_NB.pdf'));

% ===================== 1C: Fisher LDA =====================
X0e = X(y==0,:); X1e = X(y==1,:);
m0_hat = mean(X0e,1); m1_hat = mean(X1e,1);
C0_hat = cov(X0e);   C1_hat = cov(X1e);
Sw = C0_hat + C1_hat;
wlda = (Sw \ (m1_hat - m0_hat)')';
z = X*wlda';

ths_lda = linspace(min(z)-2, max(z)+2, 1200)';
pos_lda = z(y==1); neg_lda = z(y==0);
[TPR_lda, FPR_lda] = roc_from_scores(pos_lda, neg_lda, ths_lda);
PD_lda = TPR_lda; PFA_lda = FPR_lda; PM_lda = 1 - PD_lda;
perrs_lda = empirical_error(PFA_lda, PM_lda, p0, p1);
[perr_min_lda, idx_lda] = min(perrs_lda);

fprintf('Q1.C  min Pe = %.6f, FPR=%.6f, TPR=%.6f, thr=%.6f\n', ...
    perr_min_lda, PFA_lda(idx_lda), PD_lda(idx_lda), ths_lda(idx_lda));

figure; plot(FPR_lda,TPR_lda,'LineWidth',2); hold on;
plot(PFA_lda(idx_lda),PD_lda(idx_lda),'o','MarkerSize',8,'LineWidth',1.5);
xlabel('False Positive  P(D=1|L=0)'); ylabel('True Positive  P(D=1|L=1)');
title('Q1.C ROC — Fisher LDA'); grid on;

% --- add signature ---
add_signature();

saveas(gcf, fullfile('figs_Q1','Q1C_ROC_LDA.pdf'));

end