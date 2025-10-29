% NPCR_UACI_final.m
% ------------------------------------------------------------
% Medición NPCR y UACI para el cifrador real.
%
% Procedimiento interactivo (humano en el loop, pero científicamente correcto):
%
% 1. El script arma un mosaico base PlainA (con gnFoto*.png o sintético).
%    Lo guarda como PlainA.png
% 2. TE PIDE que ejecutes tu cifrador real (Image_Encrypt_Real_Plane_Con_Ataque_Brute_Force.m)
%    MANUALMENTE sobre ese mosaico, con UNA clave+sal fija.
%    Ese cifrador debe producir 02-ImaEncrypt.png
%    -> El script copia ese archivo como CipherA_tryK.png
%
% 3. El script genera PlainB = PlainA con un solo píxel modificado.
%    Lo guarda como PlainB.png
% 4. TE PIDE que vuelvas a cifrar (misma clave+sal) para ese PlainB
%    -> produce de nuevo 02-ImaEncrypt.png
%    -> El script lo copia como CipherB_tryK.png
%
% 5. Calcula NPCR y UACI entre CipherA y CipherB.
%    Repite el proceso varios trials, escogiendo distintos píxeles aleatorios.
%
% 6. Entrega media y sigma de NPCR/UACI y guarda Reporte_NPCRUACI_final.txt
%
% Nota importante: Este script no modifica tu cifrador, asume que tú
% controlas qué imagen se encripta (por ejemplo alimentando gnFoto*.png).
% Es decir, PlainA/PlainB deben ser realmente cifradas por tu rutina.
% ------------------------------------------------------------

clearvars -except -regexp '^__'; close all; clc;

%% =============== CONFIG ======================
cipher_filename = '02-ImaEncrypt.png'; % salida estándar del cifrador
tilesFolder     = pwd;                 % carpeta con gnFoto*.png
np_default      = 2;                   % mosaico np x np
makeSyntheticIfNone = true;            % crear mosaico sintético si no hay gnFoto*.png
numTrials       = 5;                   % cuántos píxeles aleatorios probamos
deltaIntensity  = 1/255;               % cambio que aplicamos a ese pixel (wrap mod 1)
outBase         = fullfile(pwd,'Resultados_NPCR_UACI_final');
timestamp       = datestr(now,'yyyy-mm-dd_HHMMSS');
outDir          = fullfile(outBase,timestamp);
if ~exist(outDir,'dir'), mkdir(outDir); end

report_path = fullfile(outDir,'Reporte_NPCRUACI_final.txt');
fid = fopen(report_path,'w');
if fid==-1
    warning('No pude abrir reporte para escritura, usando consola.');
end
log_local(sprintf('=== NPCR/UACI final ===\nGenerado: %s\n\n', datestr(now)));

%% =============== 1. Preparar mosaico base (PlainA) ===============
[tileList, tileH, tileW] = load_or_generate_tiles(tilesFolder, makeSyntheticIfNone);
if isempty(tileList)
    error('No pude obtener ni tiles reales ni sintéticos.');
end

np_answer = input(sprintf('Ingrese np mosaico (Enter = %d): ', np_default),'s');
if isempty(np_answer)
    np = np_default;
else
    np = str2double(np_answer);
    if isnan(np) || np<1
        np = np_default;
    end
end
log_local(sprintf('np = %d\n', np));

PlainA_full = build_mosaic(tileList, np);
PlainA_full = mat2gray(PlainA_full);
[H,W] = size(PlainA_full);
imwrite(PlainA_full, fullfile(outDir,'PlainA.png'));
log_local('PlainA.png guardado. Cifra ESTO con TU cifrador y misma clave+sal.\n');

%% Vectores para ir guardando resultados de cada trial
NPCR_list = zeros(numTrials,1);
UACI_list = zeros(numTrials,1);

for trial = 1:numTrials
    fprintf('--- Trial %d / %d ---\n', trial, numTrials);
    log_local(sprintf('\n--- Trial %d / %d ---\n', trial, numTrials));

    % 2. Pedir al usuario que cifre PlainA_full con su rutina
    input(sprintf(['Pausa: \n' ...
        '1) Asegúrate de que tu cifrador va a encriptar PlainA (archivo PlainA.png)\n' ...
        '   con la MISMA clave+sal que usarás en todo el experimento.\n' ...
        '2) Ejecuta manualmente Image_Encrypt_Real_Plane_Con_Ataque_Brute_Force.m\n' ...
        '   para producir %s.\n' ...
        '3) Pulsa ENTER cuando ya exista %s actualizado.\n'], ...
        cipher_filename, cipher_filename),'s');

    if ~exist(cipher_filename,'file')
        error('No se encontró %s después del cifrado de PlainA.', cipher_filename);
    end

    CipherA = im2double(imread(cipher_filename));
    if ndims(CipherA)==3, CipherA = rgb2gray(CipherA); end
    % Ajustar tamaño si hiciera falta (idealmente no)
    CipherA = imresize(CipherA,[H W]);
    imwrite(CipherA, fullfile(outDir,sprintf('CipherA_trial%02d.png',trial)));

    % 3. Generar PlainB = PlainA con 1 pixel modificado
    PlainB = PlainA_full;
    % Elegimos un pixel aleatorio:
    r = randi([1,H]); c = randi([1,W]);
    PlainB(r,c) = mod(PlainB(r,c)+deltaIntensity,1); % flip leve
    imwrite(PlainB, fullfile(outDir,sprintf('PlainB_trial%02d.png',trial)));
    log_local(sprintf('PlainB_trial%02d.png: pixel (%d,%d) perturbado.\n',trial,r,c));

    % 4. Pedir cifrar PlainB con MISMA clave+sal
    input(sprintf(['Ahora:\n' ...
        '1) Asegúrate de que tu cifrador va a encriptar PlainB_trial%02d.png\n' ...
        '   (MISMO key+salt que en PlainA).\n' ...
        '2) Ejecuta nuevamente tu cifrador para producir %s.\n' ...
        '3) Pulsa ENTER cuando ya exista el nuevo %s.\n'],...
        trial, cipher_filename, cipher_filename),'s');

    if ~exist(cipher_filename,'file')
        error('No se encontró %s después del cifrado de PlainB.', cipher_filename);
    end

    CipherB = im2double(imread(cipher_filename));
    if ndims(CipherB)==3, CipherB = rgb2gray(CipherB); end
    CipherB = imresize(CipherB,[H W]);
    imwrite(CipherB, fullfile(outDir,sprintf('CipherB_trial%02d.png',trial)));

    %% 5. Calcular NPCR y UACI entre CipherA y CipherB
    [NPCR_val, UACI_val] = compute_NPCR_UACI(CipherA, CipherB);

    NPCR_list(trial) = NPCR_val;
    UACI_list(trial) = UACI_val;

    log_local(sprintf('Trial %02d: NPCR=%.4f %% , UACI=%.4f %%\n', trial, NPCR_val*100, UACI_val*100));
end

%% 6. Estadística final
NPCR_mean = mean(NPCR_list);
NPCR_std  = std(NPCR_list);
UACI_mean = mean(UACI_list);
UACI_std  = std(UACI_list);

log_local(sprintf('\n=== RESULTADOS FINALES ===\n'));
log_local(sprintf('NPCR promedio = %.4f %% (std=%.4f %%)\n', NPCR_mean*100, NPCR_std*100));
log_local(sprintf('UACI promedio = %.4f %% (std=%.4f %%)\n', UACI_mean*100, UACI_std*100));

save(fullfile(outDir,'NPCR_UACI_results.mat'), ...
     'NPCR_list','UACI_list','NPCR_mean','NPCR_std','UACI_mean','UACI_std', ...
     'H','W','numTrials','deltaIntensity','np');

if fid~=-1
    fclose(fid);
end

fprintf('\nListo. Resultados en:\n  %s\n', outDir);


%% ===================== FUNCIONES LOCALES =====================

function [tileList, tileH, tileW] = load_or_generate_tiles(folder, allowSynthetic)
    d = dir(fullfile(folder,'gnFoto*.png'));
    tileList = {};
    if ~isempty(d)
        for k=1:length(d)
            img = im2double(imread(fullfile(d(k).folder,d(k).name)));
            if ndims(img)==3, img = rgb2gray(img); end
            tileList{end+1} = img; %#ok<AGROW>
        end
    elseif allowSynthetic
        baseSize = 512;
        t1 = rand(baseSize);                     
        [X,Y] = meshgrid(1:baseSize,1:baseSize);
        t2 = 0.5*(sin(2*pi*X/32)+1).*rand(baseSize); 
        t3 = gradient_pattern(baseSize,baseSize);
        t4 = checker_pattern(baseSize,baseSize,16);
        tileList = {t1,t2,t3,t4};
    end

    if isempty(tileList)
        tileH = []; tileW = [];
        return;
    end
    tileH = size(tileList{1},1);
    tileW = size(tileList{1},2);
    for k=1:numel(tileList)
        tileList{k} = mat2gray(tileList{k});
    end
end

function M = build_mosaic(tileList, np)
    tH = size(tileList{1},1);
    tW = size(tileList{1},2);
    H  = tH*np;
    W  = tW*np;
    M  = zeros(H,W);
    cnt=1;
    for r=1:np
        for c=1:np
            if cnt > numel(tileList)
                tile = zeros(tH,tW);
            else
                tile = tileList{cnt};
            end
            rr = (r-1)*tH + (1:tH);
            cc = (c-1)*tW + (1:tW);
            M(rr,cc) = imresize(tile,[tH tW]);
            cnt = cnt+1;
        end
    end
    M = mat2gray(M);
end

function pat = checker_pattern(H,W,blockSz)
    if nargin<3, blockSz=16; end
    [X,Y] = meshgrid(1:W,1:H);
    pat = mod(floor(X/blockSz)+floor(Y/blockSz),2);
    pat = mat2gray(pat);
end

function pat = gradient_pattern(H,W)
    [X,Y] = meshgrid(linspace(0,1,W),linspace(0,1,H));
    pat = 0.5*X + 0.5*Y;
    pat = mat2gray(pat);
end

function [NPCR_val, UACI_val] = compute_NPCR_UACI(C1, C2)
    % Asegurar double en [0,1]
    C1 = mat2gray(C1);
    C2 = mat2gray(C2);
    [H,W] = size(C1);
    if any(size(C2) ~= [H W])
        C2 = imresize(C2,[H W]);
    end

    % NPCR: porcentaje de pixeles que cambiaron
    D = (abs(C1 - C2) > 0.0005); % umbral >0 para evitar ruido numérico flotante
    NPCR_val = sum(D(:)) / numel(D);

    % UACI: promedio de |C1 - C2| / rango
    UACI_val = mean(abs(C1(:)-C2(:)));

    % Si deseas expresarlo como %, multiplica *100 al reportarlo.
end

function log_local(msg)
    persistent fidPersist
    if isempty(fidPersist)
        % leer fid del workspace base
        try
            fidPersist = evalin('base','fid');
        catch
            fidPersist = -1;
        end
    end
    fprintf('%s', msg);
    if fidPersist~=-1
        fprintf(fidPersist,'%s', msg);
    end
end
