function add_signature(name, nuid, email)

if nargin < 1 || isempty(name)
    name  = 'Atharva Prashant Kale';
end
if nargin < 2 || isempty(nuid)
    nuid  = '002442878';
end
if nargin < 3 || isempty(email)
    email = 'kale.ath@northeastern.edu';
end


ax = gca;
if isempty(ax) || ~isa(ax,'matlab.graphics.axis.Axes')
    ax = axes('Position',[0.13 0.11 0.775 0.815]); %#ok<LAXES>
end


pos = ax.Position;                % [x y w h] in normalized figure coords
deltaY  = 0.04;                   % vertical offset below axes
xRight  = pos(1) + pos(3);
yUnder  = max(pos(2) - deltaY, 0.0);

sig = sprintf('   %s\n   Northeastern ID: %s\n   Email: %s', name, nuid, email);

annotation('textbox', [xRight yUnder 0.001 0.001], ...
    'String', sig, ...
    'Units', 'normalized', ...
    'FitBoxToText', 'on', ...
    'HorizontalAlignment', 'right', ...
    'VerticalAlignment', 'bottom', ...
    'EdgeColor', 'none', ...
    'FontSize', 6, ...
    'Color', [0.3 0.3 0.3]);
end