function [CM, CM_prob, avg_risk] = q2_main_export()

rng(42,'twister');
N = 10000;
pi = [0.25 0.25 0.25 0.25];

% Parameters 
m = [-2 -2;  2 -2;  2  2; -2  2];
C(:,:,1) = [1.0  0.3;  0.3 1.0];
C(:,:,2) = [1.0 -0.2; -0.2 1.0];
C(:,:,3) = [1.2  0.0;  0.0 0.6];
C(:,:,4) = [0.8  0.2;  0.2 0.8];

% Sample data
L = randsample(4, N, true, pi);
X = zeros(N,2);
for j=1:4
    idx = (L==j);
    X(idx,:) = mvnrnd(m(j,:), C(:,:,j), nnz(idx));
end

% MAP decision (0–1 loss)
logp = zeros(N,4);
for j=1:4
    logp(:,j) = log(pi(j)) + log_gaussian_pdf(X, m(j,:), C(:,:,j));
end
[~, D_map] = max(logp, [], 2);

% Confusion (counts + row-normalized)
CM = confusionmat(L, D_map, 'Order', 1:4);
CM_prob = CM ./ sum(CM,2);
fprintf('Q2.A Confusion (counts):\n'); disp(CM);
fprintf('Q2.A Row-normalized probs:\n'); disp(CM_prob);

% ERM with asymmetric loss
Lambda = [0 10 10 100;
          1  0 10 100;
          1  1  0 100;
          1  1  1   0];
maxlog = max(logp, [], 2);
W = exp(logp - maxlog);
risk = W * Lambda.';
[~, D_erm] = min(risk, [], 2);
avg_risk = mean(arrayfun(@(n) Lambda(D_erm(n), L(n)), 1:N));
fprintf('Q2.B Average risk = %.4f\n', avg_risk);

% ---------------- Plots ----------------
if ~exist('figs_Q2','dir'), mkdir figs_Q2; end

% (1) Scatter by true class; opacity shows correctness under MAP
correct = (D_map == L);
figure('Color','w'); hold on; grid on;
markers = {'o','s','^','x'};
cols = lines(4);
for j=1:4
    idx = (L==j);
    scatter(X(idx & correct,1), X(idx & correct,2), 8, cols(j,:), 'filled', 'Marker', markers{j});
    h = scatter(X(idx & ~correct,1), X(idx & ~correct,2), 8, cols(j,:), 'Marker', markers{j});
    h.MarkerEdgeAlpha = 0.25;
end
legend({'C0 ✓','C0 ✗','C1 ✓','C1 ✗','C2 ✓','C2 ✗','C3 ✓','C3 ✗'}, 'Location','eastoutside');
title('Q2.A — Scatter by true class (higher opacity = correct)'); xlabel('x_1'); ylabel('x_2');
if exist('add_signature','file'), add_signature(); end
saveas(gcf, fullfile('figs_Q2','Q2A_scatter.pdf'));

% (2) Confusion heatmap (counts)
figure('Color','w');
imagesc(CM); axis image; colormap(parula); colorbar;
xticks(1:4); yticks(1:4);
xlabel('Decision i'); ylabel('True class j');
title('Q2.A — Confusion Matrix (counts)');
for r = 1:4
    for c = 1:4
        text(c, r, num2str(CM(r,c)), 'HorizontalAlignment','center', 'Color','k', 'FontWeight','bold');
    end
end
if exist('add_signature','file'), add_signature(); end
saveas(gcf, fullfile('figs_Q2','Q2A_confusion_heatmap.pdf'));

end