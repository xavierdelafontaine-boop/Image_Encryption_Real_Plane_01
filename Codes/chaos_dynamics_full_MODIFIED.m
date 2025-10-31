function OUT = chaos_dynamics_full(varargin)
% CHAOS_DYNAMICS_FULL — Full dynamic diagnostics for the chaotic map (Eqs. 1–2)
% - Uses (x_t, y_t) after a transient t (exactly what the cipher consumes)
% - Lyapunov (λ1, λ2) with convergence plots
% - Sensitivity to initial perturbations; finite-precision robustness (double/single/fixed)
% - Cycle detection under quantization
% - Stability maps: λ1 vs k, λ1 vs beta, and 2D map λ1(k,beta)
% - Saves all figures (PNG) and data (MAT/CSV/JSON) into 'Resultados_Chaos'
%
% Quick start:
%   OUT = chaos_dynamics_full('saveFigs',true,'saveData',true);

  %  clear all;close all;clc;


%% Default parameters
P.x0     = 0.00115125010800;
P.y0     = 0.71331112180721;
P.beta0  = 1.02166152228031;   % base beta
P.k0     = 12;                 % base k
P.t0     = 3055;               % base transient (0..9999)

P.N_main = 100000;   % samples for phase + convergence
P.N_lyap = 100000;   % length for sensitivity/precision/cycles
P.N_map  = 5000;     % iterations for maps (faster)

% Scans (edit to trade accuracy vs speed; 16 GB RAM available)
P.scanK      = 0:99;           % k-axis
P.scanTStep  = 200;            % t = 0 : step : 9999
P.betaMin    = 1.00000000000000;
P.betaMax    = 1.99999999999999;
P.nBeta1D    = 120;            % λ1 vs beta resolution
P.nBeta2D    = 60;             % beta resolution for 2D map
P.nK2D       = 50;             % k resolution for 2D map (subsample 0..99)

% Sensitivity / finite precision
P.delta0         = 1e-12;
P.precFixedScale = 2^18;       % fixed-point grid = 1/scale

% Output
P.saveFigs = true;
P.saveData = true;
P.outDir   = 'Resultados_Chaos';
P.seed     = 1;

% Equilibria (for base k0, beta0)
P.eq_guesses = linspace(-pi, pi, 25);

P = parse_args(P, varargin{:});
rng(P.seed);
if ~exist(P.outDir,'dir'), mkdir(P.outDir); end

% Scan axes
SCAN.k_values = P.scanK(:)';
SCAN.t_values = 0:P.scanTStep:9999; if SCAN.t_values(end)~=9999, SCAN.t_values(end+1)=9999; end
SCAN.beta1D   = linspace(P.betaMin, P.betaMax, P.nBeta1D);
SCAN.k2D      = round(linspace(0, 99, max(2,P.nK2D)));
SCAN.beta2D   = linspace(P.betaMin, P.betaMax, P.nBeta2D);

%% Transient (x_t, y_t) and main trajectory
fprintf('=== Base point: k=%d, t=%d, beta=%.14f ===\n', P.k0, P.t0, P.beta0);
[x_t, y_t] = apply_transient(P.x0, P.y0, P.k0, P.beta0, P.t0);
fprintf('State after transient: x_t=%.12g, y_t=%.12g\n', x_t, y_t);

[xs, ys] = iterate_from(x_t, y_t, P.k0, P.beta0, P.N_main); 

% Phase portrait — robust density
figure('Color','w');
phase_density_plot(xs, ys);

% Agregar título
title(sprintf('Phase-space density (k=%d, t=%d, \\beta=%.14f)', ...
    P.k0, P.t0, P.beta0));

% Ajustes de fuente (solo ejes)
ax = gca;
ax.FontSize = 11;            % números pequeños
ax.XLabel.FontSize = 12;     % etiquetas más grandes
ax.YLabel.FontSize = 12;
ax.Title.FontSize = 12;      % opcional
ax.XLabel.FontWeight = 'bold';
ax.YLabel.FontWeight = 'bold';

% Añadir la barra de color
cb = colorbar;
cb.Label.String = 'Points density';
cb.Label.FontSize = 13;
cb.Label.FontWeight = 'bold';
cb.Label.Rotation = 90;

% Guardar figura
save_fig('phase_density', P);


%% Lyapunov (λ1, λ2) + convergence

LE = lyapunov_QR(xs, ys, P.k0, P.beta0); 
figure('Color','w','Position',[100 100 450 300]);
plot_lyapunov_convergence(LE);

% Ajuste del eje principal
ax = gca;
ax.Position = [0.12 0.12 0.78 0.75];

% Buscar y mover el recuadro "Final zoom"
ax_all = findall(gcf, 'Type', 'axes');
if numel(ax_all) > 1
    zoom_ax = ax_all(1);               % asume que el primero es el recuadro pequeño
    zoom_ax.Position = [0.3 0.5 0.25 0.31]; % centrado y más pequeño
    zoom_ax.FontSize = 9;              % opcional: ajustar fuente dentro del recuadro
end

save_fig('lyapunov_convergence', P);
fprintf('Lyapunov: λ1=%.6f, λ2=%.6f, sum=%.3e\n', LE.lambda1, LE.lambda2, LE.lambda1+LE.lambda2);

%% Sensitivity and finite-precision robustness

Sens = sensitivity_CI(x_t, y_t, P.k0, P.beta0, P.N_lyap, P.delta0); 
figure('Color','w');
plot_sensitivity(Sens);

% Ajustes de límites y ejes
ax = gca;
xlim([0 1e5]);                    % recorta a las iteraciones relevantes
%ylim([1e-2 1e5]);                 % evita la zona muerta inferior
ylim([1 1e5]);   
ax.XTick = 0:2e4:1e5;
ax.YTick = [1e-2 1e0 1e2 1e4];
ax.YTickLabel = {'10^{-2}','1','10^{2}','10^{4}'};

% Estilo del eje
ax.FontSize = 12;
ax.LineWidth = 1.2;
ax.YScale = 'log';                % asegura escala logarítmica vertical
xlabel('n');
ylabel('||\Delta_n||','Interpreter','tex');

% Guardar figura
save_fig('sensitivity_to_initial_perturbation', P);

Prec = precision_study(x_t, y_t, P.k0, P.beta0, P.N_lyap, P.precFixedScale);
figure('Color','w');
plot_precision_summary(Prec);
save_fig('finite_precision_study', P);


%% Cycle detection (repetitiveness) under quantization
Cyc = detect_cycles_quant(x_t, y_t, P.k0, P.beta0, P.N_lyap, P.precFixedScale);
figure('Color','w');
plot_cycles(Cyc, P);
save_fig('cycle_detection', P);

%% Stability maps (chaotic vs non-chaotic regions)
lambda_vs_k    = lyap_vs_k(P, SCAN);           % λ1 vs k (beta=beta0, t=t0)
lambda_vs_beta = lyap_vs_beta(P, SCAN);        % λ1 vs beta (k=k0, t=t0)
LAM_kb         = lyap_map_k_beta(P, SCAN);     % λ1(k,beta) at fixed t0

figure('Color','w'); plot_lambda_vs_k(SCAN.k_values, lambda_vs_k, P);       save_fig('lambda_vs_k', P);
figure('Color','w'); plot_lambda_vs_beta(SCAN.beta1D, lambda_vs_beta, P);   save_fig('lambda_vs_beta', P);
% figure('Color','w'); plot_lambda_map_kb(LAM_kb, SCAN.k2D, SCAN.beta2D, P);  save_fig('lambda_map_k_beta', P);
 
% === Generar mapa 2D (plano tipo calor) ===
figure('Color','w');
plot_lambda_map_kb(LAM_kb, SCAN.k2D, SCAN.beta2D, P, '2d');
save_fig('lambda_map_k_beta_2d', P);

% === Generar superficie 3D ===
figure('Color','w');
plot_lambda_map_kb(LAM_kb, SCAN.k2D, SCAN.beta2D, P, '3d');
save_fig('lambda_map_k_beta_3d', P);


%% Equilibria and linear classification at (k0, beta0)
[eq_list, class_list] = equilibrium_points_and_class(P.k0, P.beta0, P.eq_guesses);
figure('Color','w');
plot_equilibria(eq_list, class_list, P);
save_fig('equilibria_classification', P);

%% Save data (MAT/CSV/JSON summary for later ChatGPT analysis)
OUT = struct();
OUT.params = P;
OUT.base   = struct('xt',x_t,'yt',y_t,'LE',LE,'Sens',Sens,'Prec',Prec,'Cyc',Cyc);
OUT.lambda_vs_k = table(SCAN.k_values(:), lambda_vs_k(:), 'VariableNames', {'k','lambda1'});
OUT.lambda_vs_beta = table(SCAN.beta1D(:), lambda_vs_beta(:), 'VariableNames', {'beta','lambda1'});
OUT.lambda_map_kb = LAM_kb;
OUT.equilibria = struct('xstar', eq_list, 'ystar', -0.5*P.k0*sin(eq_list), 'class', {class_list});

if P.saveData
    save(fullfile(P.outDir, 'Resultados_Completos.mat'), 'OUT', '-v7.3');
    writematrix([SCAN.k_values(:) lambda_vs_k(:)], fullfile(P.outDir,'lambda_vs_k.csv'));
    writematrix([SCAN.beta1D(:)   lambda_vs_beta(:)], fullfile(P.outDir,'lambda_vs_beta.csv'));
    % Lightweight JSON summary for later upload to ChatGPT
% Lightweight JSON summary for later upload to ChatGPT
summary = struct();
summary.xt = x_t;
summary.yt = y_t;
summary.lambda1 = LE.lambda1;
summary.lambda2 = LE.lambda2;
summary.lambda_sum = LE.lambda1 + LE.lambda2;
summary.sensitivity_slope = Sens.slope;
summary.delta0 = P.delta0;
summary.fixed_grid = 1 / P.precFixedScale;
summary.cycle_found = Cyc.found;

% Add figure names only
fig_files = dir(fullfile(P.outDir, '*.png'));
summary.figures = {fig_files.name}';

% Use MATLAB built-in JSON encoder (more robust)
json_text = jsonencode(summary);

% Save JSON safely
fid = fopen(fullfile(P.outDir, 'summary_for_chatgpt.json'), 'w');
if fid ~= -1
    fwrite(fid, json_text, 'char');
    fclose(fid);
    fprintf('Saved summary: %s\n', fullfile(P.outDir, 'summary_for_chatgpt.json'));
else
    warning('Could not write summary_for_chatgpt.json');
end
end
fprintf('\nDone. Figures and data saved to "%s"\n', P.outDir);
end

%% ================= Core functions =================
function [xt, yt] = apply_transient(x0, y0, k, beta, t)
xt = x0; yt = y0;
for i = 1:t, [xt, yt] = map_step(xt, yt, k, beta); end
end

function [xs, ys] = iterate_from(x0, y0, k, beta, N)
xs = zeros(N,1); ys = zeros(N,1); xt = x0; yt = y0;
for n=1:N, [xt, yt] = map_step(xt, yt, k, beta); xs(n)=xt; ys(n)=yt; end
end

function [x1, y1] = map_step(x, y, k, beta)
cb = cos(beta); sb = sin(beta);
xy = y + k*sin(x);
x1 = x*cb - xy*sb;
y1 = x*sb + xy*cb;
end

function J = jacobian_at(x,~,k,beta)
cb = cos(beta); sb = sin(beta);
J = [ cb - (k*cos(x))*sb,  -sb;
      sb + (k*cos(x))*cb,   cb ];
end

function LE = lyapunov_QR(xs, ys, k, beta)
N = numel(xs);
Q = eye(2); acc = zeros(2,1);
lambda1_hist = zeros(N,1); lambda2_hist = zeros(N,1);
x = xs(1); y = ys(1);
for n=1:N
    J = jacobian_at(x, y, k, beta);
    Z = J * Q; [Q, R] = qr(Z, 0);
    rn = abs(diag(R)); acc = acc + log(rn + eps);
    lambda1_hist(n) = acc(1)/n; lambda2_hist(n) = acc(2)/n;
    x = xs(n); y = ys(n);
end
LE.lambda1 = acc(1)/N; LE.lambda2 = acc(2)/N;
LE.lambda1_hist = lambda1_hist; LE.lambda2_hist = lambda2_hist;
LE.iter = (1:N)';
end

%% ============== Sensitivity / Precision / Cycles =================
function Sens = sensitivity_CI(xt, yt, k, beta, N, delta0)
x1=xt; y1=yt; x2=xt+delta0; y2=yt; dist = zeros(N,1);
for n=1:N
    [x1,y1] = map_step(x1,y1,k,beta);
    [x2,y2] = map_step(x2,y2,k,beta);
    dist(n) = hypot(x1-x2, y1-y2);
end
idx = 1:max(10, round(N/3));
pp = polyfit(idx', log(dist(idx)+eps), 1);
Sens.slope = pp(1); Sens.iter=(1:N)'; Sens.dist=dist; Sens.delta0=delta0;
end

function Prec = precision_study(xt, yt, k, beta, N, scale)
modes = {'double','single','fixed'}; traj = struct();
for i=1:numel(modes)
    mode = modes{i}; [xs,ys] = iterate_precision(xt, yt, k, beta, N, mode, scale);
    traj.(mode) = struct('xs',xs,'ys',ys);
end
LEd = lyapunov_QR(traj.double.xs, double(traj.double.ys), k, beta);
LEs = lyapunov_QR(double(traj.single.xs), double(traj.single.ys), k, beta);
LEf = lyapunov_QR(double(traj.fixed.xs),  double(traj.fixed.ys),  k, beta);
dd_single = hypot(traj.double.xs - double(traj.single.xs), traj.double.ys - double(traj.single.ys));
dd_fixed  = hypot(traj.double.xs - double(traj.fixed.xs),  traj.double.ys - double(traj.fixed.ys));
Prec.LE = struct('double',LEd,'single',LEs,'fixed',LEf);
Prec.div_single = dd_single; Prec.div_fixed = dd_fixed; Prec.iter=(1:N)'; Prec.scale=scale;
end

function [xs, ys] = iterate_precision(x, y, k, beta, N, mode, scale)
xs=zeros(N,1); ys=zeros(N,1);
switch mode
    case 'double'
        for n=1:N, [x,y]=map_step(x,y,k,beta); xs(n)=x; ys(n)=y; end
    case 'single'
        x=single(x); y=single(y);
        for n=1:N, [x,y]=map_step(x,y,k,beta); x=single(x); y=single(y); xs(n)=double(x); ys(n)=double(y); end
    case 'fixed'
        x=quant_to_grid(double(x),scale); y=quant_to_grid(double(y),scale);
        for n=1:N, [x,y]=map_step(x,y,k,beta); x=quant_to_grid(double(x),scale); y=quant_to_grid(double(y),scale);
            xs(n)=x; ys(n)=y; end
end
end

function v=quant_to_grid(z,scale), v=round(z*scale)/scale; end

function Cyc = detect_cycles_quant(xt, yt, k, beta, N, scale)
Cyc.grid = 1/scale; Cyc.found=false; Cyc.period=NaN; Cyc.hitIndex=NaN;
x=quant_to_grid(xt,scale); y=quant_to_grid(yt,scale);
seen = containers.Map('KeyType','char','ValueType','int32');
for n=1:N
    [x,y] = map_step(x,y,k,beta);
    x = quant_to_grid(x,scale); y = quant_to_grid(y,scale);
    key = sprintf('%.0f,%.0f', round(x*scale), round(y*scale));
    if isKey(seen,key), Cyc.found=true; first=seen(key); Cyc.period=n-first; Cyc.hitIndex=n; break;
    else, seen(key)=n; end
end
end

%% =================== Lyapunov maps ===================
function lambda = lyap_vs_k(P, SCAN)
lambda = zeros(numel(SCAN.k_values),1);
for i=1:numel(SCAN.k_values)
    k = SCAN.k_values(i);
    lambda(i) = lyap_scalar_from_transient(P.x0,P.y0,k,P.beta0,P.t0,P.N_map);
end
end

function lambda = lyap_vs_beta(P, SCAN)
lambda = zeros(numel(SCAN.beta1D),1);
for i=1:numel(SCAN.beta1D)
    b = SCAN.beta1D(i);
    lambda(i) = lyap_scalar_from_transient(P.x0,P.y0,P.k0,b,P.t0,P.N_map);
end
end

function LAM = lyap_map_k_beta(P, SCAN)
LAM = zeros(numel(SCAN.beta2D), numel(SCAN.k2D)); % rows: beta, cols: k
for ib=1:numel(SCAN.beta2D)
    bcur = SCAN.beta2D(ib);
    for ik=1:numel(SCAN.k2D)
        kcur = SCAN.k2D(ik);
        LAM(ib,ik) = lyap_scalar_from_transient(P.x0,P.y0,kcur,bcur,P.t0,P.N_map);
    end
end
end

function lambda = lyap_scalar_from_transient(x0,y0,k,beta,t,N)
for i=1:t, [x0,y0]=map_step(x0,y0,k,beta); end
[xs,ys] = iterate_from(x0,y0,k,beta,N);
LE = lyapunov_QR(xs,ys,k,beta);
lambda = LE.lambda1;
end

%% =================== Plot helpers (English UI) ===================
function phase_density_plot(xs, ys)
mask = isfinite(xs) & isfinite(ys); xs = xs(mask); ys = ys(mask);
if isempty(xs), warning('No valid data for phase plot.'); return; end
nbins = 400;
xedges = linspace(min(xs), max(xs), nbins);
yedges = linspace(min(ys), max(ys), nbins);
C = histcounts2(xs, ys, xedges, yedges);
imagesc(xedges, yedges, log(C' + 1)); axis xy; axis tight; safe_colormap(); colorbar;
xlabel('x_n'); ylabel('y_n'); title('Phase-space density'); grid on;
end

function plot_lyapunov_convergence(LE)
plot(LE.iter, LE.lambda1_hist, 'r', 'LineWidth', 1); hold on;
plot(LE.iter, LE.lambda2_hist, 'b', 'LineWidth', 1);
w = max(200, round(numel(LE.iter)/200));
lam1_s = movmean(LE.lambda1_hist, w);
lam2_s = movmean(LE.lambda2_hist, w);
plot(LE.iter, lam1_s, 'Color',[0.8 0 0], 'LineWidth', 1.8);
plot(LE.iter, lam2_s, 'Color',[0 0 0.8], 'LineWidth', 1.8);
yline(LE.lambda1, '--r'); yline(LE.lambda2, '--b');
xlabel('Iteration n'); ylabel('\lambda estimate');
legend('\lambda_1','\lambda_2','\lambda_1 (smoothed)','\lambda_2 (smoothed)','Location','best');
title(sprintf('Convergence: \\lambda_1=%.4f, \\lambda_2=%.4f, sum=%.2e', LE.lambda1, LE.lambda2, LE.lambda1+LE.lambda2));
grid on;
axes('Position',[.58 .55 .32 .32]); box on; hold on;
N = numel(LE.iter); m = min(5000, N-10);
plot(LE.iter(end-m:end), lam1_s(end-m:end), 'r', 'LineWidth', 1.2);
plot(LE.iter(end-m:end), lam2_s(end-m:end), 'b', 'LineWidth', 1.2);
title('Final zoom'); set(gca,'FontSize',8);
end

function plot_sensitivity(S)
semilogy(S.iter, S.dist, 'k'); grid on;
xlabel('n'); ylabel('||\Delta_n||');
title(sprintf('Sensitivity to initial perturbation (\\delta_0=%.1e), slope≈%.3g', S.delta0, S.slope));
end

function plot_precision_summary(Prec)
subplot(2,1,1);
vals = [Prec.LE.double.lambda1, Prec.LE.double.lambda2;
        Prec.LE.single.lambda1, Prec.LE.single.lambda2;
        Prec.LE.fixed.lambda1,  Prec.LE.fixed.lambda2];
bar(vals); grid on; xticklabels({'double','single','fixed'});
ylabel('\lambda'); title('Lyapunov exponents by precision'); legend('\lambda_1','\lambda_2');

subplot(2,1,2);
semilogy(Prec.iter, Prec.div_single, 'r'); hold on;
semilogy(Prec.iter, Prec.div_fixed,  'b'); grid on;
xlabel('n'); ylabel('||deviation vs double||');
title(sprintf('Divergence vs double (fixed grid=%.1e)', 1/Prec.scale));
legend('single','fixed');
end

%%_______________________________________________________________

function plot_cycles(Cyc, P)
axis off;
if Cyc.found
    txt = sprintf('Cycle detected.\nperiod=%d (grid=%.3g)', Cyc.period, 1/P.precFixedScale);
else
    txt = sprintf('No cycle detected (N=%d, grid=%.3g)', P.N_lyap, 1/P.precFixedScale);
end
text(0.1,0.6,txt,'Interpreter','none','FontSize',12);
end

function plot_lambda_vs_k(k, lambda, ~)
    plot(k, lambda, 'k.-', 'LineWidth', 1.2);
    grid on; box on;

    % === Etiquetas y título ===
    xlabel('k', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('\lambda_1', 'FontSize', 14, 'FontWeight', 'bold');
    title('\lambda_1 vs k', 'FontSize', 16, 'FontWeight', 'bold');

    % === Control del tamaño de fuente de los números y ejes ===
    ax = gca;
    ax.FontSize = 12;     % tamaño de los números en los ejes
    ax.LineWidth = 1.2;   % grosor de ejes
end


function plot_lambda_vs_beta(beta, lambda, ~)
    plot(beta, lambda, 'k.-', 'LineWidth', 1.2);
    grid on; box on;

    % === Etiquetas y título ===
    xlabel('\beta', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('\lambda_1', 'FontSize', 14, 'FontWeight', 'bold');
    title('\lambda_1 vs \beta', 'FontSize', 16, 'FontWeight', 'bold');

    % === Control del tamaño de fuente de los números y ejes ===
    ax = gca;
    ax.FontSize = 12;
    ax.LineWidth = 1.2;
end


function plot_lambda_map_kb(LAM, K, B, ~, mode)
% === λ₁(k,β) Surface or Heatmap ===
% LAM: matriz (numel(B) x numel(K)) con λ₁
% K:   vector de k (columnas)
% B:   vector de β (filas)
if nargin < 5, mode = '3d'; end

set(gcf,'Renderer','opengl');  % asegurar render 3D

switch lower(mode)
    case '3d'
   [KK, BB] = meshgrid(K, B);   % malla para surf     
        surf(KK, BB, LAM, ...
     'EdgeColor', 'none', ...
     'FaceColor', 'interp', ...
     'FaceLighting', 'gouraud');
      shading interp; axis tight; box on; grid on;      
    
% === CONTROL DE COLOR Y BRILLO ===
colormap(jet(512));                         % 'jet' tiene rojos más vivos
vmin = min(LAM(:));
vmax = max(LAM(:));
range_scale = 0.7;                          % amplía el rango un 20 % para forzar el rojo
caxis([vmin vmax*range_scale]);             % extiende el rango para mostrar el rojo

% Aclara la paleta (corrige gamma)
cmap = colormap;
cmap = cmap.^0.7;                           % <1 aclara todos los tonos
colormap(cmap);

cb = colorbar;
cb.Label.String = '\lambda_1 chaos intensity';
cb.Label.FontSize = 14;
cb.Label.FontWeight = 'bold';
cb.Label.Rotation = 90;

% === ILUMINACIÓN: para que el rojo brille ===
lighting gouraud;                           % iluminación suave
material shiny;                             % superficie reflectante
camlight headlight;                         % fuente de luz frontal
         
      
      % === CONTROL DE COLOR: realce del rojo brillante ===
        view(45, 28);                % ángulo agradable
        xlabel('k', 'FontSize', 14, 'FontWeight', 'bold');
        ylabel('\beta', 'FontSize', 14, 'FontWeight', 'bold');
        zlabel('\lambda_1', 'FontSize', 14, 'FontWeight', 'bold');
        title('3D Lyapunov surface \lambda_1(k,\beta)', ...
              'FontSize', 16, 'FontWeight', 'bold');

        ax = gca;
        ax.FontSize = 12;
        ax.LineWidth = 1.2;
        camlight headlight; material dull;

    case '2d'
        imagesc(K, B, LAM); 
        set(gca, 'YDir', 'normal'); 
        axis tight; box on; grid on;
        safe_colormap();
        cb = colorbar;
        cb.Label.String = '\lambda_1 Chaos intensity';
        cb.Label.FontSize = 14;
        cb.Label.FontWeight = 'bold';
        cb.Label.Rotation = 90;
        xlabel('k', 'FontSize', 14, 'FontWeight', 'bold');
        ylabel('\beta', 'FontSize', 14, 'FontWeight', 'bold');
        title('\lambda_1(k,\beta)', 'FontSize', 16, 'FontWeight', 'bold');

    otherwise
        error('mode must be ''3d'' or ''2d''');
end
end



function [eq_list, class_list] = equilibrium_points_and_class(k, beta, guesses)
% Solve x + C sin x = 0 ; C = k*sin(beta)/(2*(1 - cos(beta)))
a = 1 - cos(beta); sb = sin(beta);
C = k*sb/(2*(a + eps));
roots_found = [];
for g = guesses(:)'
    try
        xstar = fzero(@(x) x + C*sin(x), g);
        xstar = wrapToPi(xstar);
        if isempty(roots_found) || all(abs(angdiff(roots_found - xstar)) > 1e-6)
            roots_found(end+1,1) = xstar; %#ok<AGROW>
        end
    catch, end
end
roots_found = unique(round(roots_found, 12));
eq_list = roots_found(:)'; class_list = cell(size(eq_list));
for i=1:numel(eq_list)
    xeq = eq_list(i); yeq = -0.5*k*sin(xeq);
    J = jacobian_at(xeq, yeq, k, beta);
    tr = trace(J);
    if abs(tr) < 2, class_list{i} = 'elliptic';
    elseif abs(tr) > 2, class_list{i} = 'hyperbolic';
    else,               class_list{i} = 'parabolic';
    end
end
end

function plot_equilibria(eq_list, class_list, P)
hold on; grid on;
for i=1:numel(eq_list)
    xeq = eq_list(i); yeq = -0.5*P.k0*sin(xeq);
    mkr = 'ko'; switch class_list{i}
        case 'elliptic',   mkr='go';
        case 'hyperbolic', mkr='rx';
        case 'parabolic',  mkr='ks';
    end
    plot(xeq, yeq, mkr, 'MarkerSize',10,'LineWidth',1.2);
    text(xeq, yeq, sprintf('  #%d (%s)', i, class_list{i}), 'Interpreter','none');
end
xlabel('x^*'); ylabel('y^*'); title(sprintf('Equilibria (k=%d, \\beta=%.14f)', P.k0, P.beta0));
end


%% ================= Utilities =================
function safe_colormap()
if exist('turbo','file'), colormap(turbo); else, colormap(jet); end
end

function save_fig(name, P)
if ~P.saveFigs, return; end
fname = fullfile(P.outDir, [name '.png']);
drawnow; try, saveas(gcf, fname); fprintf('Saved figure: %s\n', fname);
catch, warning('Could not save figure %s', fname); end
end

function P = parse_args(P, varargin)
if mod(numel(varargin),2)~=0, error('Args must be name-value pairs.'); end
for i=1:2:numel(varargin)
    name = varargin{i}; val = varargin{i+1};
    if isfield(P, name), P.(name) = val; else, error('Unknown parameter: %s', name); end
end
end

function y = wrapToPi(x), y = mod(x + pi, 2*pi) - pi; end
function d = angdiff(a), d = atan2(sin(a), cos(a)); end

% --- tiny JSON saver (no toolbox required) ---
function savejson(path, S)
fid = fopen(path,'w'); if fid<0, warning('Cannot write %s',path); return; end
c = onCleanup(@() fclose(fid));
fprintf(fid, '{\n');
keys = fieldnames(S);
for i=1:numel(keys)
    k = keys{i}; v = S.(k);
    fprintf(fid, '  "%s": %s', k, json_val(v));
    if i < numel(keys), fprintf(fid, ','); end
    fprintf(fid, '\n');
end
fprintf(fid, '}\n');
end

function s = json_val(v)
if ischar(v), s = ['"' strrep(v,'"','\"') '"']; return; end
if islogical(v), s = ternary(v,'true','false'); return; end
if isnumeric(v)
    if isscalar(v), s = sprintf('%.16g', v);
    else
        s = '['; for i=1:numel(v), s=[s sprintf('%.16g',v(i))]; if i<numel(v), s=[s ',']; end, end; s=[s ']'];
    end, return;
end
if isstruct(v)
    f = fieldnames(v);
    s = '{'; for i=1:numel(f)
        key=f{i}; val=json_val(v.(key));
        s=[s '"' key '": ' val]; if i<numel(f), s=[s ', ']; end
    end, s=[s '}']; return;
end
s = 'null';
end

function o = ternary(cond,a,b), if cond, o=a; else, o=b; end, end
