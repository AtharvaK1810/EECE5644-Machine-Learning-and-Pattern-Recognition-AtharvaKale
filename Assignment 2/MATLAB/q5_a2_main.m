function q5_a2_main(name,nuid,email)

rng(5);
K  = 4;               % number of categories
N  = 100;             % sample size
theta_true = [0.10 0.20 0.50 0.20];

% --------- Simulate categorical labels (1..K) ----------
Z  = randsample(1:K, N, true, theta_true);
Nk = histcounts(Z, 1:K+1);          
Ntot = sum(Nk);

% --------- ML estimate ----------
theta_ML = Nk / Ntot;

% --------- MAP estimates ----------
alphas = { [1 1 1 1], [10 10 10 10] };   
theta_MAP = zeros(numel(alphas), K);
for i = 1:numel(alphas)
    a = alphas{i};
    
    theta_MAP(i,:) = (Nk + a - 1) / (Ntot + sum(a) - K);
end

% --------- Figure (bar chart) ----------
if ~exist('figs_A2','dir'), mkdir figs_A2; end
fh = figure('Color','w','Position',[100 100 900 420]);

% Order: True, ML, MAP(α=1), MAP(α=10)
plot_matrix = [theta_true; theta_ML; theta_MAP(1,:); theta_MAP(2,:)];
bh = bar(plot_matrix, 'grouped');

grid on; box on;
xlabel('Category k');
ylabel('\theta_k');
title('A2–Q5: Categorical–Dirichlet Estimates');

% X tick labels as categories 1..K
set(gca,'XTick',1:size(plot_matrix,1), ...
        'XTickLabel',{'True','ML','MAP(\alpha=1)','MAP(\alpha=10)'}, ...
        'FontSize',11);

% Legend mapped to categories (θ1..θ4)
lg = legend({'\theta_1','\theta_2','\theta_3','\theta_4'}, ...
            'Location','northoutside','Orientation','horizontal');
set(lg,'Interpreter','tex');

% Optional small y-limit padding
ylim([0 max(plot_matrix(:))*1.15]);

%  signature
if exist('add_signature','file') == 2
    try
        add_signature(name,nuid,email);  
    catch
        
    end
end

saveas(fh, 'figs_A2/Q5_Categorical_Dirichlet.pdf');

% --------- Results to CSV ----------
if ~exist('results','dir'), mkdir results; end
rowLabels = {'True','ML','MAP_a1','MAP_a10'};
T = array2table(plot_matrix, ...
    'VariableNames', {'theta1','theta2','theta3','theta4'}, ...
    'RowNames', rowLabels);
writetable(T, 'results/Q5_theta_estimates.csv', 'WriteRowNames', true);

% Also echo to command window for quick verification
disp('Q5: Parameter estimates (rows: True, ML, MAP_a1, MAP_a10)');
disp(T);

end
