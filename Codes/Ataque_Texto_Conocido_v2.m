% Ataque_Texto_Conocido_v2_fixed.m
% Known-plaintext attack diagnostics (robusto frente a tiles de distinto tamaño)
% Saves figures and Reporte_AtaqueTextoConocido.txt
%
% Usage: run in folder with gnFoto1..gnFoto4.png and/or 02-ImaEncrypt.png; 
% requires KeySeed_KDF_DualOut, CPMfun, and/or your encryptor if needed.

clc; clear; close all;

outDir = fullfile(pwd,'Resultados_AtaqueTextoConocido');
if ~exist(outDir,'dir'), mkdir(outDir); end

reportFile = fullfile(outDir,'Reporte_AtaqueTextoConocido.txt');
fid = fopen(reportFile,'w');

fprintf(fid,'Known-plaintext attack report\nGenerated: %s\n\n',datestr(now));
fprintf('Known-plaintext attack diagnostics — results folder: %s\n', outDir);

% --- 1) Load plaintexts (gnFoto1..4 -> build mosaic) ---
nFaces = 4;
faceNames = arrayfun(@(k) sprintf('gnFoto%d.png',k), 1:nFaces, 'uni',false);

plainTiles = cell(1,nFaces);
found = false(1,nFaces);
for k=1:nFaces
    if exist(faceNames{k},'file')
        I = imread(faceNames{k});
        if size(I,3)==3, I = rgb2gray(I); end
        I = im2double(I);
        plainTiles{k} = I;
        found(k) = true;
    else
        fprintf('Warning: plaintext tile %s not found.\n', faceNames{k});
        fprintf(fid,'Warning: plaintext tile %s not found.\n', faceNames{k});
    end
end

if ~all(found)
    fprintf('ERROR: missing one or more plaintext tiles (gnFoto1..4). Aborting.\n');
    fprintf(fid,'ERROR: missing plaintext tiles. Aborting.\n');
    fclose(fid);
    return;
end

% --- 1.b) Check sizes and normalize to common tile size (smallest detected) ---
sizes = zeros(nFaces,2);
for k=1:nFaces
    [h,w] = size(plainTiles{k});
    sizes(k,:) = [h,w];
end
minH = min(sizes(:,1));
minW = min(sizes(:,2));

if any(sizes(:,1)~=minH) || any(sizes(:,2)~=minW)
    fprintf('Note: Not all tiles have same size. Rescaling all tiles to %d x %d (smallest found).\n', minH, minW);
    fprintf(fid,'Rescaling tiles to %d x %d (smallest found)\n', minH, minW);
    for k=1:nFaces
        if ~isequal(size(plainTiles{k}), [minH minW])
            plainTiles{k} = imresize(plainTiles{k}, [minH minW], 'bilinear');
        end
    end
end

% assemble mosaic 2x2 
M_h = minH; M_w = minW;
mosaic = zeros(2*M_h, 2*M_w);
cnt = 1;
for row = 1:2
    for col = 1:2
        r1 = (row-1)*M_h+1; r2 = row*M_h;
        c1 = (col-1)*M_w+1; c2 = col*M_w;
        mosaic(r1:r2, c1:c2) = plainTiles{cnt};
        cnt = cnt + 1;
    end
end
Plain = mosaic;
imwrite(mat2gray(Plain), fullfile(outDir,'Plain_Mosaic.png'));

% --- 2) Load ciphertext (prefer existing 02-ImaEncrypt.png) ---
cipherName = '02-ImaEncrypt.png';
if ~exist(cipherName,'file')
    fprintf('Cipher image %s not found.\n', cipherName);
    fprintf('Option: run your encryptor to produce it? (y/N): ');
    resp = input('','s');
    if strcmpi(resp,'y')
        try
            fprintf('Running user encryptor script (will ask for key/salt) ...\n');
            Image_Encrypt_Real_Plane_Con_Ataque_Brute_Force; %#ok<NASGU,STOUT>
        catch ME
            fprintf('Failed to run encryptor: %s\n', ME.message);
            fprintf(fid,'Failed to run encryptor: %s\n', ME.message);
            fclose(fid); return;
        end
        if ~exist(cipherName,'file')
            fprintf('Encryptor ran but %s still not found. Aborting.\n', cipherName);
            fprintf(fid,'Encryptor ran but %s not created. Aborting.\n', cipherName);
            fclose(fid); return;
        end
    else
        fprintf('Aborting — supply %s in working folder or run encryptor manually.\n', cipherName);
        fprintf(fid,'Aborting — cipher file not provided.\n'); fclose(fid); return;
    end
end

Cipher = im2double(imread(cipherName));
if size(Cipher,3)==3, Cipher = rgb2gray(Cipher); end

% If cipher size differs from assembled Plain, attempt to resize cipher to match Plain:
[ph,pw] = size(Plain);
[ch,cw] = size(Cipher);
if ch ~= ph || cw ~= pw
    fprintf('Cipher size (%d x %d) differs from assembled Plain (%d x %d). Resizing cipher to match Plain.\n', ch, cw, ph, pw);
    fprintf(fid,'Resizing cipher from %d x %d to %d x %d to match plain.\n', ch, cw, ph, pw);
    Cipher = imresize(Cipher, [ph pw], 'bilinear');
end

imwrite(mat2gray(Cipher), fullfile(outDir,'Cipher.png'));

% --- 3) Basic stats and correlation ---
P = Plain(:);
C = Cipher(:);
R = corrcoef(P,C);
pearson = R(1,2);
nmse = mean((P-C).^2) / var(P);
fprintf('Pearson corr(Plain,Cipher) = %.6f\n', pearson);
fprintf(fid,'Pearson corr(Plain,Cipher) = %.6f\n', pearson);
fprintf('Normalized MSE (MSE/var(Plain)) = %.6e\n', nmse);
fprintf(fid,'Normalized MSE = %.6e\n\n', nmse);

% --- 4) Estimate mask M = mod(C - P, 1)  (images in [0,1]) ---
Mask = mod(Cipher - Plain, 1);
imwrite(mat2gray(Mask), fullfile(outDir,'Estimated_Mask.png'));
fprintf('Saved Estimated_Mask.png\n');
fprintf(fid,'Estimated mask saved as Estimated_Mask.png\n');

% mask histogram
figure('Visible','off'); imhist(im2uint8(mat2gray(Mask))); title('Mask histogram');
saveas(gcf, fullfile(outDir,'Mask_histogram.png')); close(gcf);
fprintf(fid,'Mask histogram saved.\n');

% --- 5) Mask autocorrelation and PSD ---
AC = xcorr2(Mask - mean(Mask(:)));
ACnorm = AC / max(abs(AC(:)));
figure('Visible','off'); imagesc(ACnorm); axis image; colorbar;
title('Mask autocorrelation (normalized)');
saveas(gcf, fullfile(outDir,'Mask_autocorr.png')); close(gcf);
fprintf(fid,'Mask autocorrelation saved.\n');

F = fftshift(abs(fft2(Mask)).^2);
Flog = log(1 + F);
figure('Visible','off'); imagesc(Flog); axis image; colorbar;
title('Mask power spectrum (log)');
saveas(gcf, fullfile(outDir,'Mask_PSD_log.png')); close(gcf);
fprintf(fid,'Mask PSD saved.\n');

% --- 6) Local correlation map (sliding window) ---
w = min(64, floor(min(size(Plain))/8)); step = floor(w/2);
[mx,my] = size(Plain);
corrMap = zeros( floor((mx-w)/step)+1, floor((my-w)/step)+1 );
ix = 1;
for i=1:step:(mx-w+1)
    iy = 1;
    for j=1:step:(my-w+1)
        blockP = Plain(i:i+w-1, j:j+w-1);
        blockC = Cipher(i:i+w-1, j:j+w-1);
        r = corrcoef(blockP(:), blockC(:));
        corrMap(ix,iy) = r(1,2);
        iy = iy + 1;
    end
    ix = ix + 1;
end
figure('Visible','off'); imagesc(corrMap); colorbar; title('Local Pearson corr (sliding window)');
saveas(gcf, fullfile(outDir,'Local_corr_map.png')); close(gcf);
fprintf(fid,'Local correlation map saved.\n');

% --- 7) Interpretive metrics: mask entropy, mask mean, mask sparsity ---
maskMean = mean(Mask(:));
maskStd  = std(Mask(:));
maskEntropy = shannon_entropy(im2uint8(mat2gray(Mask)));
sparsity = sum(abs(Mask(:))>1e-6)/numel(Mask);
fprintf('Mask mean=%.6e std=%.6e entropy(byte)=%g sparsity=%g\n', maskMean, maskStd, maskEntropy, sparsity);
fprintf(fid,'Mask mean=%.6e std=%.6e entropy(byte)=%g sparsity=%g\n', maskMean, maskStd, maskEntropy, sparsity);

% --- 8) Save variables for later analysis ---
save(fullfile(outDir,'AtaqueTextoConocido_results.mat'),'Plain','Cipher','Mask','AC','F','corrMap');

% --- 9) Short textual conclusions (automatic) ---
fprintf(fid,'\n--- Automated conclusions ---\n');
if pearson < 0.1
    fprintf(fid,'Low global Pearson correlation (%.4f) suggests large decorrelation between plain and cipher.\n', pearson);
else
    fprintf(fid,'Non-negligible correlation (Pearson=%.4f) indicates some leakage of plaintext structure to ciphertext.\n', pearson);
end
if maskEntropy < 7
    fprintf(fid,'Mask byte-entropy is low (%.2f bits/byte) -> mask may have structure / weak randomness.\n', maskEntropy);
else
    fprintf(fid,'Mask byte-entropy is high (%.2f bits/byte) -> mask behaves closer to random.\n', maskEntropy);
end
fprintf(fid,'Full results and images saved under folder: %s\n', outDir);
fclose(fid);

fprintf('Done. Report saved to %s\n', reportFile);

%% ===== helpers =====
function H = shannon_entropy(u8)
% approximate Shannon entropy (bits) for uint8 image
p = histcounts(u8,0:256,'Normalization','probability');
p(p==0)=[]; H = -sum(p .* log2(p));
end
