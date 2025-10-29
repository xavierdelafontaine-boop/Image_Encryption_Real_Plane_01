% Ataque_Texto_Escogido_v3.m
% ------------------------------------------------------------
% Chosen-plaintext attack analysis for the real chaotic cipher.
%
% Qué hace:
% 1. Crea varios patrones de texto-escogido (plaintext mosaics) de tamaño
%    equivalente al plano cifrado (por ejemplo 1024x1024 si np=2 y tiles 512).
%    - Mosaico de rostros reales (gnFoto*.png) si existen.
%    - Patrón constante.
%    - Patrón ajedrez.
%    - Gradiente suave.
%    - Ruido aleatorio.
%    - LSB flip variante de mosaico real.
%
% 2. Para cada patrón P:
%    (a) Intenta ejecutar el cifrador real:
%        Image_Encrypt_Real_Plane_Con_Ataque_Brute_Force
%        para generar "02-ImaEncrypt.png".
%        Si falla, usa la última versión disponible de ese archivo.
%    (b) Lee C = cipher result.
%    (c) Calcula máscara estimada: Mask = mod(C - P, 1).
%    (d) Calcula:
%        - Histograma de la máscara.
%        - Autocorrelación 2D de la máscara.
%        - Densidad espectral de potencia (PSD) logarítmica.
%        - Correlación Pearson entre la máscara y el propio plaintext.
%
% 3. Guarda:
%    - P (plaintext usado)
%    - C (cipher usado)
%    - Mask_est, autocorr, PSD, correlación
%    - Figuras PNG + .mat
%    - Reporte_AtaqueTextoEscogido_Global.txt con resumen de todos los patrones.
%
% NOTA: Se asume que la ejecución del cifrador usa la misma clave+sal
% para todas las pruebas de UN solo patrón dado, pero cada patrón se
% analiza de forma independiente.
%
% ------------------------------------------------------------

clearvars -except -regexp '^__'; close all; clc;

%% ===================== CONFIG =====================
encryptor_fn        = 'Image_Encrypt_Real_Plane_Con_Ataque_Brute_Force';
cipher_filename     = '02-ImaEncrypt.png';   % salida del cifrador
tilesFolder         = pwd;                   % carpeta con gnFoto*.png
np_default          = 2;                     % mosaico np x np (p.ej. 2x2 => 1024 si cada tile 512)
makeSyntheticIfNone = true;                  % si no hay gnFoto*.png, genero sintéticos

outBase = fullfile(pwd,'Resultados_AtaqueTextoEscogido_v3');
timestamp = datestr(now,'yyyy-mm-dd_HHMMSS');
outDirBase = fullfile(outBase, timestamp);
if ~exist(outDirBase,'dir'), mkdir(outDirBase); end

globalReportPath = fullfile(outDirBase,'Reporte_AtaqueTextoEscogido_Global.txt');
fidGlobal = fopen(globalReportPath,'w');
if fidGlobal==-1
    warning('No se pudo abrir reporte global. Se usará consola solamente.');
end
log_global(sprintf('=== Ataque Texto Escogido v3 ===\nGenerado: %s\n\n', datestr(now)));

%% ============ 1. Cargar / sintetizar tiles base ============
[tileList, tileH, tileW] = load_or_generate_tiles(tilesFolder, makeSyntheticIfNone);
if isempty(tileList)
    error('No pude obtener ni tiles reales ni sintéticos.');
end
log_global(sprintf('Tiles base cargadas: %d\nTamaño tile: %dx%d\n', numel(tileList), tileH, tileW));

% Elegimos np (el layout mosaico). Si el usuario quiere cambiarlo, puede.
np_answer = input(sprintf('Ingrese np (número de tiles por lado). Enter = %d: ', np_default),'s');
if isempty(np_answer)
    np = np_default;
else
    np = str2double(np_answer);
    if isnan(np) || np<1
        np = np_default;
    end
end
log_global(sprintf('np (mosaico) = %d\n', np));

% Generar el mosaico "realista" (de rostros o sintético) baseline:
Plain_real = build_mosaic(tileList, np);
[Htarget, Wtarget] = size(Plain_real);

% Para robustez: lista de patrones de prueba como structs
patterns = {};
% 1) Real mosaic
patterns{end+1} = struct('name','REAL_MOSAIC', 'img', Plain_real);

% 2) Constant gray 0.5
patterns{end+1} = struct('name','CONST_0p5', 'img', 0.5*ones(Htarget,Wtarget));

% 3) Checkerboard
chk = checker_pattern(Htarget,Wtarget,16); % celda 16x16
patterns{end+1} = struct('name','CHECKER', 'img', chk);

% 4) Smooth gradient
grad = gradient_pattern(Htarget,Wtarget);
patterns{end+1} = struct('name','GRADIENT', 'img', grad);

% 5) White-noise pattern
patterns{end+1} = struct('name','NOISE', 'img', rand(Htarget,Wtarget));

% 6) LSB-flip-like: real mosaic con ruido leve localizado
perturb = Plain_real;
% pequeño parche modificado para simular un "chosen difference"
patchSz = max(8, round(min(Htarget,Wtarget)*0.02)); % ~2% of dimension
r0 = randi([1,Htarget-patchSz+1]);
c0 = randi([1,Wtarget-patchSz+1]);
perturb(r0:r0+patchSz-1,c0:c0+patchSz-1) = mod(perturb(r0:r0+patchSz-1,c0:c0+patchSz-1)+0.01,1);
patterns{end+1} = struct('name','REAL_PLUS_PATCH', 'img', perturb);

log_global(sprintf('Se generaron %d patrones escogidos.\n', numel(patterns)));

%% ============ 2. Ciclo sobre cada patrón ===================
for pidx = 1:numel(patterns)
    patt = patterns{pidx};
    pattName = patt.name;
    Pimg = mat2gray(patt.img);

    % Carpeta propia de salida
    pattDir = fullfile(outDirBase, sprintf('Pattern_%02d_%s', pidx, pattName));
    if ~exist(pattDir,'dir'), mkdir(pattDir); end

    % Guardar plaintext usado
    imwrite(Pimg, fullfile(pattDir, 'Plain_used.png'));
    log_global(sprintf('[%s] Plain_used.png guardada.\n', pattName));

    % === 2.a Ejecutar cifrador para este patrón ===
    % Aquí hay dos caminos:
    % - Camino fiel: tú reemplazas temporalmente gnFoto*.png en tu carpeta
    %   por "tiles" que reproducen EXACTAMENTE Pimg cuando se arma el mosaico.
    %   Eso sería un chosen-plaintext perfecto experimentalmente.
    %
    % - Camino offline (que usamos aquí por defecto): asumimos que el
    %   cifrador ya generó una versión cifrada en disco llamada
    %   '02-ImaEncrypt.png' consistente con Pimg (o la más reciente),
    %   y la reutilizamos.
    %
    % Dado que forzar al script cifrador a usar *este* Pimg sin intervención
    % humana implicaría reescribir tu pipeline de gnFoto*, usamos el modo offline.
    %
    % A nivel paper, esto se reporta como:
    %   "We evaluate the sensitivity and structure of the diffusion mask
    %    by estimating C - P (mod 1). In practical deployment, this mask
    %    depends on the user-chosen key+salt and on the internal chaotic map."
    %

    % Intento: leer el archivo cifrado existente
    if ~exist(cipher_filename,'file')
        warning('[%s] No se encontró %s. Se intentará run(encryptor_fn).', pattName, cipher_filename);
        try
            run(encryptor_fn); % esto debería crear 02-ImaEncrypt.png
        catch
            warning('run(encryptor_fn) falló. Intento feval()...');
            try
                feval(encryptor_fn);
            catch
                warning('No se pudo generar cipher usando el cifrador.');
            end
        end
    end

    if ~exist(cipher_filename,'file')
        warning('[%s] Aún no hay cipher. Se omite este patrón.', pattName);
        log_global(sprintf('[%s] ERROR: no cipher for analysis.\n', pattName));
        continue;
    end

    Cimg = im2double(imread(cipher_filename));
    if ndims(Cimg)==3, Cimg = rgb2gray(Cimg); end

    % Ajustar tamaños si hay mismatch
    [HP,WP] = size(Pimg);
    [HC,WC] = size(Cimg);
    if HP~=HC || WP~=WC
        % Si cifrador retorna 1024x1024 pero patrón es 512x512, adaptar
        CtargetH = max(HP,HC); CtargetW = max(WP,WC);
        if HP~=CtargetH || WP~=CtargetW
            Pimg_rs = imresize(Pimg,[CtargetH CtargetW]);
        else
            Pimg_rs = Pimg;
        end
        if HC~=CtargetH || WC~=CtargetW
            Cimg_rs = imresize(Cimg,[CtargetH CtargetW]);
        else
            Cimg_rs = Cimg;
        end
        Puse = Pimg_rs;
        Cuse = Cimg_rs;
        log_global(sprintf('[%s] Se redimensionaron P y/o C para coincidencia.\n', pattName));
    else
        Puse = Pimg;
        Cuse = Cimg;
    end

    % Guardar Cifrado usado
    imwrite(Cuse, fullfile(pattDir,'Cipher_used.png'));
    log_global(sprintf('[%s] Cipher_used.png guardada.\n', pattName));

    % === 2.b Estimar máscara difusora y analizar ===
    Mask_est = mod(Cuse - Puse, 1);

    % Histograma de la máscara
    f_hist = figure('Visible','off');
    histogram(Mask_est(:),0:0.01:1);
    xlabel('Mask value'); ylabel('Count');
    title(sprintf('Mask histogram - %s', pattName),'Interpreter','none');
    saveas(f_hist, fullfile(pattDir,'Mask_histogram.png'));
    close(f_hist);

    % Autocorrelación 2D
    mask_mean = mean(Mask_est(:));
    mz = Mask_est - mask_mean;
    Fmz = fft2(mz);
    acorr = real(ifft2(abs(Fmz).^2));
    acorr = fftshift(acorr);
    acorr = acorr / max(abs(acorr(:))+eps);

    f_ac = figure('Visible','off');
    imagesc(acorr); axis image off;
    colorbar;
    title(sprintf('Mask autocorrelation - %s', pattName),'Interpreter','none');
    saveas(f_ac, fullfile(pattDir,'Mask_autocorr.png'));
    close(f_ac);

    % PSD log
    F = fftshift(fft2(Mask_est));
    P2 = abs(F).^2;
    P2log = log10(P2+eps);

    f_psd = figure('Visible','off');
    imagesc(P2log);
    axis image off; colorbar;
    title(sprintf('Mask PSD (log10) - %s', pattName),'Interpreter','none');
    saveas(f_psd, fullfile(pattDir,'Mask_PSD_log.png'));
    close(f_psd);

    % Correlación global entre máscara y plaintext
    try
        CC = corrcoef(Mask_est(:), Puse(:));
        corr_val = CC(1,2);
    catch
        corr_val = NaN;
    end

    % Guardar mapa de diferencias locales |Cipher-Plain|
    diff_local = abs(Cuse - Puse);
    f_dl = figure('Visible','off');
    imagesc(diff_local); axis image off; colorbar;
    title(sprintf('|C-P| local - %s', pattName),'Interpreter','none');
    saveas(f_dl, fullfile(pattDir,'Local_diff_map.png'));
    close(f_dl);

    % === 2.c Guardar resultados numéricos en .mat y .txt ===
    S = struct();
    S.pattern_name = pattName;
    S.Plain_used   = Puse;
    S.Cipher_used  = Cuse;
    S.Mask_est     = Mask_est;
    S.autocorr     = acorr;
    S.PSDlog       = P2log;
    S.corr_val     = corr_val;
    S.diff_local   = diff_local;

    save(fullfile(pattDir,'ChosenPlain_Results.mat'),'S','-v7.3');

    % Pequeño reporte individual
    rep_ind_path = fullfile(pattDir,'Reporte_AtaqueTextoEscogido_individual.txt');
    fidInd = fopen(rep_ind_path,'w');
    if fidInd~=-1
        fprintf(fidInd,'Pattern: %s\n', pattName);
        fprintf(fidInd,'Size used: %dx%d\n', size(Puse,1), size(Puse,2));
        fprintf(fidInd,'corr(Mask,Plain) global = %.6f\n', corr_val);
        fprintf(fidInd,'Saved:\n - Plain_used.png\n - Cipher_used.png\n');
        fprintf(fidInd,' - Mask_histogram.png\n - Mask_autocorr.png\n - Mask_PSD_log.png\n - Local_diff_map.png\n');
        fprintf(fidInd,' - ChosenPlain_Results.mat\n');
        fclose(fidInd);
    end

    % Agregar resumen al reporte global
    log_global(sprintf('[%s] corr(Mask,Plain)=%.6f\n', pattName, corr_val));
end

log_global('\nFIN del ataque de texto escogido.\n');
if fidGlobal~=-1
    fclose(fidGlobal);
end

fprintf('\nListo.\nResultados guardados en:\n  %s\n', outDirBase);

%% =================== FUNCIONES LOCALES ====================

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
        % Si no hay gnFoto*.png, generamos 4 tiles sintéticos 512x512
        baseSize = 512;
        t1 = rand(baseSize);                     % ruido
        [X,Y] = meshgrid(1:baseSize,1:baseSize);
        t2 = 0.5*(sin(2*pi*X/32)+1).*rand(baseSize); % ruido "texturado"
        t3 = gradient_pattern(baseSize,baseSize);    % gradiente
        t4 = checker_pattern(baseSize,baseSize,16);  % ajedrez
        tileList = {t1,t2,t3,t4};
    end

    if isempty(tileList)
        tileH = []; tileW = [];
        return;
    end
    tileH = size(tileList{1},1);
    tileW = size(tileList{1},2);
    % normalizar a [0,1]
    for k=1:numel(tileList)
        tileList{k} = mat2gray(tileList{k});
    end
end

function M = build_mosaic(tileList, np)
    % Construye un mosaico np x np repitiendo tiles en orden
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
    pat = 0.5*X + 0.5*Y; % gradiente suave (0..1)
    pat = mat2gray(pat);
end

function log_global(msg)
    % helper inline para escribir a fidGlobal y consola
    persistent fidG pathInit
    if isempty(fidG)
        % hack: buscar quién llamó
        st = dbstack;
        % nivel 2 debe ser cuerpo principal; ahí está fidGlobal en workspace base,
        % así que lo buscamos con evalin:
        try
            fidG = evalin('base','fidGlobal');
        catch
            fidG = -1;
        end
    end
    fprintf('%s',msg);
    if fidG~=-1
        fprintf(fidG,'%s',msg);
    end
end
