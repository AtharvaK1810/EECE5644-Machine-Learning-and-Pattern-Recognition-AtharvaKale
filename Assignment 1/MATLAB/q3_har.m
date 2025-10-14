function q3_har()


root = '/Users/atharva/Documents/MS in IoT/Sem 3/Machine Learning and Pattern Recognition/Assignment 1/Datasets/UCI HAR Dataset';

X_train = readmatrix(fullfile(root,'train','X_train.txt'));
y_train = readmatrix(fullfile(root,'train','y_train.txt'));
X_test  = readmatrix(fullfile(root,'test','X_test.txt'));
y_test  = readmatrix(fullfile(root,'test','y_test.txt'));

X = [X_train; X_test];
y = [y_train; y_test];

lam = 1e-4;
model = fit_gaussian_generative(X, y, lam);
[yhat, ~] = predict_gaussian_generative(model, X);

CM = confusionmat(y, yhat);
acc = mean(y==yhat);
fprintf('HAR: Accuracy=%.4f, Error=%.4f\n', acc, 1-acc);
disp('Confusion matrix:'); disp(CM);

[coeff,score] = pca(X);
figure; scatter(score(:,1), score(:,2), 4, 'filled'); grid on;
title('HAR — PCA(2) scatter'); xlabel('PC1'); ylabel('PC2');
add_signature();
if ~exist('figs_Q3','dir'), mkdir figs_Q3; end
saveas(gcf, fullfile('figs_Q3','Q3_HAR_PCA2.pdf'));
end