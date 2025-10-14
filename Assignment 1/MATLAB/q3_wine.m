function q3_wine()

csvPath = '/Users/atharva/Documents/MS in IoT/Sem 3/Machine Learning and Pattern Recognition/Assignment 1/Datasets/wine+quality/winequality-white.csv';
T = readtable(csvPath, 'Delimiter',';');
y = T.quality; T.quality = [];
X = table2array(T);

lam = 1e-6;
model = fit_gaussian_generative(X, y, lam);
[yhat, ~] = predict_gaussian_generative(model, X);

CM = confusionmat(y, yhat);
acc = mean(y==yhat);
fprintf('Wine Quality: Accuracy=%.4f, Error=%.4f\n', acc, 1-acc);
disp('Confusion matrix:'); disp(CM);

[coeff,score] = pca(X);
figure; scatter(score(:,1), score(:,2), 8, 'filled'); grid on;
title('Wine Quality — PCA(2) scatter'); xlabel('PC1'); ylabel('PC2');
add_signature();
if ~exist('figs_Q3','dir'), mkdir figs_Q3; end
saveas(gcf, fullfile('figs_Q3','Q3_wine_PCA2.pdf'));
end