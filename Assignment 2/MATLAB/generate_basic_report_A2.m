function generate_basic_report_A2()

clc;

resDir  = 'results';
figDir  = 'figs_A2';
outFile = fullfile(resDir,'A2_basic_report.txt');

if ~exist(resDir,'dir'), error('results/ folder not found. Run run_all_A2 first.'); end
if ~exist(figDir,'dir'), warning('figs_A2/ not found. Figure list may be empty.'); end

% ---- Read CSVs ----
q1_minPe = safeReadScalar(fullfile(resDir,'Q1_MAP_minPe.csv'));
q1_logerr = safeReadMatrix(fullfile(resDir,'Q1_logistic_errors.csv')); % [3x2] [lin quad]

q2_ml_mse = safeReadScalar(fullfile(resDir,'Q2_ML_MSE.csv'));
q2_gamma_mse = safeReadMatrix(fullfile(resDir,'Q2_MAP_gamma_MSE.csv')); % [N x 2] (gamma, mse)
bestGamma = NaN; bestMapMSE = NaN;
if ~isempty(q2_gamma_mse)
    [bestMapMSE, idx] = min(q2_gamma_mse(:,2));
    bestGamma = q2_gamma_mse(idx,1);
end

q5_theta = safeReadMatrix(fullfile(resDir,'Q5_theta_estimates.csv')); % rows: [true; ML; MAPs...]

% ---- Enumerate figures ----
figList = {};
if exist(figDir,'dir')
    d = dir(fullfile(figDir,'*.pdf'));
    [~,ix] = sort({d.name}); d = d(ix);
    figList = {d.name}';
end

% ---- Write the report ----
fid = fopen(outFile,'w');
fprintf(fid,'Assignment 2 – Basic Report (Plain Text)\n');
fprintf(fid,'Author: Atharva Prashant Kale | NUID: 002442878 | kale.ath@northeastern.edu\n');
fprintf(fid,'Generated on: %s\n\n', datestr(now));

% -------- Q1 --------
fprintf(fid,'Q1. Two-Class Classification (Gaussian Mixtures)\n');
if ~isnan(q1_minPe)
    fprintf(fid,'- MAP (true mixture) minimum probability of error (validation): %.4f\n', q1_minPe);
else
    fprintf(fid,'- MAP result: (missing CSV results/Q1_MAP_minPe.csv)\n');
end
if ~isempty(q1_logerr) && size(q1_logerr,2) >= 2
    linErr  = q1_logerr(:,1);
    quadErr = q1_logerr(:,2);
    fprintf(fid,'- Logistic Linear P(error) [D50, D500, D5000]: ');
    fprintf(fid,'%.4f ', linErr); fprintf(fid,'\n');
    fprintf(fid,'- Logistic Quadratic P(error) [D50, D500, D5000]: ');
    fprintf(fid,'%.4f ', quadErr); fprintf(fid,'\n');
else
    fprintf(fid,'- Logistic errors CSV missing or malformed (expected [3x2]).\n');
end
fprintf(fid,'Simple interpretation: MAP uses the exact pdf and is stronger; quadratic features help logistic converge as data grows.\n\n');

% -------- Q2 --------
fprintf(fid,'Q2. Cubic Regression (ML vs MAP ridge)\n');
if ~isnan(q2_ml_mse)
    fprintf(fid,'- ML MSE (validation): %.6f\n', q2_ml_mse);
else
    fprintf(fid,'- ML MSE missing (results/Q2_ML_MSE.csv)\n');
end
if ~isnan(bestMapMSE)
    fprintf(fid,'- Best MAP MSE (validation): %.6f at gamma = %.3g\n', bestMapMSE, bestGamma);
else
    fprintf(fid,'- MAP gamma sweep missing (results/Q2_MAP_gamma_MSE.csv)\n');
end
fprintf(fid,'Simple interpretation: MAP adds L2 regularization; best gamma balances bias and variance.\n\n');

% -------- Q3 --------
fprintf(fid,'Q3. Localization MAP (Contours)\n');
fprintf(fid,'- Contour plots saved as Q3_contours_K1..K4.pdf under figs_A2/.\n');
fprintf(fid,'Simple interpretation: combining Gaussian prior with range measurements creates bowl-shaped objective; contours shrink around the true location as K increases.\n\n');

% -------- Q4 --------
fprintf(fid,'Q4. Bayes Boundary (DHS 2.13 illustration)\n');
fprintf(fid,'- Plot: figs_A2/Q4_BayesBoundary.pdf\n');
fprintf(fid,'Simple interpretation: with unequal covariances/priors, the Bayes boundary is generally quadratic.\n\n');

% -------- Q5 --------
fprintf(fid,'Q5. Categorical–Dirichlet (ML & MAP)\n');
if ~isempty(q5_theta)
    fprintf(fid,'- Theta estimates (rows = True; ML; then MAP for listed alphas):\n');
    dispToFile(fid, q5_theta);
else
    fprintf(fid,'- Theta estimates CSV missing (results/Q5_theta_estimates.csv)\n');
end
fprintf(fid,'Simple interpretation: ML = counts/total; MAP shrinks toward prior (stronger when alpha is larger).\n\n');

% -------- Figures list --------
fprintf(fid,'Figure files found in figs_A2/:\n');
if ~isempty(figList)
    for i=1:numel(figList)
        fprintf(fid,'- %s\n', figList{i});
    end
else
    fprintf(fid,'(no PDFs found — run run_all_A2 first)\n');
end

fclose(fid);
fprintf('Wrote plain report: %s\n', outFile);
end

% ---------- helpers ----------
function x = safeReadScalar(p)
if exist(p,'file')
    try
        x = readmatrix(p);
        x = x(1);
    catch
        x = NaN;
    end
else
    x = NaN;
end
end

function M = safeReadMatrix(p)
if exist(p,'file')
    try
        M = readmatrix(p);
    catch
        M = [];
    end
else
    M = [];
end
end

function dispToFile(fid, M)
for r=1:size(M,1)
    for c=1:size(M,2)
        if c>1, fprintf(fid,'\t'); end
        fprintf(fid,'%.6f', M(r,c));
    end
    fprintf(fid,'\n');
end
end
