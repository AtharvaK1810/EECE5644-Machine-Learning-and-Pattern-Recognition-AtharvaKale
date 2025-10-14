clear; clc;

% ===== PATHS =====
WINE_CSV = '/Users/atharva/Documents/MS in IoT/Sem 3/Machine Learning and Pattern Recognition/Assignment 1/Datasets/wine+quality/winequality-white.csv';
HAR_ROOT = '/Users/atharva/Documents/MS in IoT/Sem 3/Machine Learning and Pattern Recognition/Assignment 1/Datasets/UCI HAR Dataset';
% =================

if ~exist('results','dir'), mkdir results; end

fprintf('=== EECE5644 A1: RUN ALL (plots + CSV) ===\n');

% --- Q1 ---
try
    fprintf('\n[Q1] Running q1_main.m ...\n');
    q1_main();
catch ME
    warning('[Q1] Failed: %s', ME.message);
end

% --- Q2 ---
try
    fprintf('\n[Q2] Running q2_main_export() (plots + CSV data) ...\n');
    [CM, CM_prob, avg_risk] = q2_main_export();    % also saves figs in figs_Q2/
    if ~exist('results','dir'), mkdir results; end
    writematrix(CM,       fullfile('results','Q2_confusion_counts.csv'));
    writematrix(CM_prob,  fullfile('results','Q2_confusion_prob.csv'));
    writematrix(avg_risk, fullfile('results','Q2_avg_risk.csv'));
catch ME
    warning('[Q2] Failed: %s', ME.message);
end

% --- Q3: Wine ---
try
    fprintf('\n[Q3/Wine] Running q3_wine_export(...) ...\n');
    [CMw, accw] = q3_wine_export(WINE_CSV);        % also saves PCA fig in figs_Q3/
    writematrix(CMw,  fullfile('results','Q3_wine_confusion.csv'));
    writematrix(accw, fullfile('results','Q3_wine_accuracy.csv'));
catch ME
    warning('[Q3/Wine] Failed: %s', ME.message);
end

% --- Q3: HAR ---
try
    fprintf('\n[Q3/HAR] Running q3_har_export(...) ...\n');
    [CMh, acch] = q3_har_export(HAR_ROOT);         % also saves PCA fig in figs_Q3/
    writematrix(CMh,  fullfile('results','Q3_har_confusion.csv'));
    writematrix(acch, fullfile('results','Q3_har_accuracy.csv'));
catch ME
    warning('[Q3/HAR] Failed: %s', ME.message);
end

fprintf('\n=== DONE. Check figs_Q1/, figs_Q2/, figs_Q3/ for PDFs and results/ for CSVs. ===\n');

% -------- Helper export wrappers --------

function [CM, acc] = q3_wine_export(csvPath)
    T = readtable(csvPath, 'Delimiter',';');
    y = T.quality; T.quality = [];
    X = table2array(T);
    lam = 1e-6;
    model = fit_gaussian_generative(X, y, lam);
    [yhat, ~] = predict_gaussian_generative(model, X);
    CM = confusionmat(y, yhat);
    acc = mean(y==yhat);
    fprintf('Wine Quality: Accuracy=%.4f, Error=%.4f\n', acc, 1-acc);

    % PCA(2) figure
    [coeff,score] = pca(X);
    if ~exist('figs_Q3','dir'), mkdir figs_Q3; end
    figure('Color','w'); scatter(score(:,1), score(:,2), 8, 'filled'); grid on;
    title('Wine Quality — PCA(2) scatter'); xlabel('PC1'); ylabel('PC2');
    if exist('add_signature','file'), add_signature(); end
    saveas(gcf, fullfile('figs_Q3','Q3_wine_PCA2.pdf'));
end

function [CM, acc] = q3_har_export(root)
    X_train = readmatrix(fullfile(root,'train','X_train.txt'));
    y_train = readmatrix(fullfile(root,'train','y_train.txt'));
    X_test  = readmatrix(fullfile(root,'test','X_test.txt'));
    y_test  = readmatrix(fullfile(root,'test','y_test.txt'));

    X = [X_train; X_test]; y = [y_train; y_test];

    lam = 1e-4;  % HAR is high-dim; small ridge stabilizes covariances
    model = fit_gaussian_generative(X, y, lam);
    [yhat, ~] = predict_gaussian_generative(model, X);
    CM = confusionmat(y, yhat);
    acc = mean(y==yhat);
    fprintf('HAR: Accuracy=%.4f, Error=%.4f\n', acc, 1-acc);

    % PCA(2) figure
    [coeff,score] = pca(X);
    if ~exist('figs_Q3','dir'), mkdir figs_Q3; end
    figure('Color','w'); scatter(score(:,1), score(:,2), 4, 'filled'); grid on;
    title('HAR — PCA(2) scatter'); xlabel('PC1'); ylabel('PC2');
    if exist('add_signature','file'), add_signature(); end
    saveas(gcf, fullfile('figs_Q3','Q3_HAR_PCA2.pdf'));
end