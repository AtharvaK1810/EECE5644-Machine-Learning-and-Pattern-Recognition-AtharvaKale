function run_all_A2()
clc; clear; close all;
addpath(genpath('.'));

% ---- signature ----
name  = 'Atharva Prashant Kale';
nuid  = '002442878';
email = 'kale.ath@northeastern.edu';

if ~exist('results','dir'), mkdir results; end
if ~exist('figs_A2','dir'), mkdir figs_A2; end

fprintf('=== EECE5644 Assignment 2: RUN ALL (Q1–Q5) ===\n');

% ---------- Q1 ----------
fprintf('\n[Q1] Two-class Gaussian mixtures (MAP + Logistic Lin/Quad)...\n');
q1 = q1_a2_main(name,nuid,email);

% ---------- Q2 ----------
fprintf('\n[Q2] Cubic regression (ML vs MAP over gamma)...\n');
q2 = q2_a2_main(name,nuid,email);

% ---------- Q3 ----------
fprintf('\n[Q3] 2D localization MAP (K = 1..4 contours)...\n');
q3_a2_main(name,nuid,email); %#ok<NASGU>

% ---------- Q4 ----------
fprintf('\n[Q4] DHS 2.13 Bayes boundary illustration...\n');
q4_a2_main(name,nuid,email);

% ---------- Q5 ----------
fprintf('\n[Q5] Categorical–Dirichlet ML & MAP estimators...\n');
q5_a2_main(name,nuid,email);

% ---------- Plain text summary ----------
fid = fopen(fullfile('results','A2_results_summary.txt'),'w');
fprintf(fid,'EECE5644 – Assignment 2 (plain summary)\n');
fprintf(fid,'Author: %s | %s | %s\n\n', name, nuid, email);

fprintf(fid,'Q1: Mixture-Gaussian + Logistic models\n');
fprintf(fid,'- MAP min P(error) (validation): %.4f\n', q1.pe_map);
fprintf(fid,'- Logistic Linear P(error) [D50/D500/D5000]: %.4f / %.4f / %.4f\n', q1.pe_lin(1), q1.pe_lin(2), q1.pe_lin(3));
fprintf(fid,'- Logistic Quadratic P(error) [D50/D500/D5000]: %.4f / %.4f / %.4f\n\n', q1.pe_quad(1), q1.pe_quad(2), q1.pe_quad(3));

fprintf(fid,'Q2: Cubic Regression\n');
fprintf(fid,'- ML MSE (validation): %.6f\n', q2.mse_ml);
fprintf(fid,'- Best MAP MSE (validation): %.6f at gamma=%.3g\n\n', q2.best_mse_map, q2.best_gamma);

fprintf(fid,'Q3: See figs_A2/Q3_contours_K*.pdf (signed)\n\n');

fprintf(fid,'Q4: See figs_A2/Q4_BayesBoundary.pdf (signed)\n\n');

fprintf(fid,'Q5: See figs_A2/Q5_Categorical_Dirichlet.pdf and results/Q5_theta_estimates.csv\n');
fclose(fid);

fprintf('\n== DONE. Figures -> figs_A2/, CSVs + summary -> results/ ==\n');
end