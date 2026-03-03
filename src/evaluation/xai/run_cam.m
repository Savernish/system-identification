% src/evaluation/xai/run_cam.m
clc; clear; close all;

script_dir = fileparts(mfilename('fullpath'));
project_root = fullfile(script_dir, '..', '..', '..');
addpath(genpath(fullfile(project_root, 'src')));

%% 1. Eğitilmiş Modeli Yükle
model_path = fullfile(project_root, 'results', 'multimodal_poc_net.mat');
if ~exist(model_path, 'file')
    error('Eğitilmiş model bulunamadı!');
end

load_data = load(model_path, 'net');
net = load_data.net; 

%% 2. Test Edilecek Verilerin Dağılımını Ayarla (6 Underdamped, 2 Overdamped)
% İstenen sınıfları bir listeye koyuyoruz
target_classes = {'Underdamped', 'Underdamped', 'Underdamped', 'Underdamped', 'Underdamped', 'Underdamped', ...
                  'Overdamped', 'Overdamped'};

% Ekranda hep aynı sırada çıkmasınlar diye listeyi rastgele karıştırıyoruz
target_classes = target_classes(randperm(8));

% Şov için devasa bir figür penceresi aç (2 satır, 4 sütun için uygun boyut)
figure('Color', 'k', 'Position', [100, 100, 1600, 800]);
sgtitle('Multimodal Model - Toplu Test ve Açıklanabilirlik (XAI)', 'FontSize', 18, 'FontWeight', 'bold');

fprintf('Toplu test başlıyor. Modeli 8 rastgele veri ile sınıyoruz...\n\n');

%% 3. Döngü İçinde 8 Veriyi Test Et
for i = 1:8
    class_name = target_classes{i};
    
    % Sınıf klasöründen rastgele dosya seç
    img_folder = fullfile(project_root, 'data', 'images', 'train', class_name);
    files = dir(fullfile(img_folder, '*.png'));
    num_samples = length(files);
    rand_sample_idx = randi(num_samples);
    
    img_path = fullfile(img_folder, files(rand_sample_idx).name);
    [~, name_only, ~] = fileparts(files(rand_sample_idx).name);
    sig_path = fullfile(project_root, 'data', 'signals', 'train', class_name, [name_only '.mat']);
    
    % 3.1 Görüntüyü Oku
    img = imread(img_path);
    if ismatrix(img)
        img = reshape(img, size(img,1), size(img,2), 1);
    end
    img_single = im2single(img);
    
    % 3.2 Sinyali Oku
    data = load(sig_path);
    sig = double(data.y(:))';
    
    % 3.3 Datastore Pipeline (Kurşun Geçirmez Formatımız)
    dsImg = transform(arrayDatastore({img_single}, 'IterationDimension', 1), @(x) x{1});
    dsSig = transform(arrayDatastore({sig}, 'IterationDimension', 1), @(x) x{1});
    
    if strcmp(net.Layers(1).Name, 'input_2D')
        dsTest = combine(dsImg, dsSig);
    else
        dsTest = combine(dsSig, dsImg);
    end
    
    % =========================================================
    % 4. MODEL TAHMİNİ (PREDICTION)
    % =========================================================
    [YPred, scores] = classify(net, dsTest);
    pred_class = char(YPred(1)); % Tahmin edilen sınıfı stringe çevir
    
    % =========================================================
    % 5. ISI HARİTASI (XAI - GRAD-CAM)
    % =========================================================
    featureLayer = 'relu_2d_2'; 
    actMap = activations(net, dsTest, featureLayer, 'OutputAs', 'channels');
    if iscell(actMap)
        actMap = actMap{1};
    end
    
    heatMap = mean(actMap, 3);
    heatMap = imresize(heatMap, [228, 448]);
    heatMap = mat2gray(heatMap);
    
    % =========================================================
    % 6. GÖRSELLEŞTİRME (2x4 Grid)
    % =========================================================
    subplot(2, 4, i);
    imshow(img);
    hold on;
    h = imagesc(heatMap);
    colormap(gca, jet); % Her subplot'un kendi renk haritası olsun
    set(h, 'AlphaData', 0.5); 
    hold off;
    
    % Doğru bildiyse YEŞİL, yanlış bildiyse KIRMIZI başlık
    if strcmp(class_name, pred_class)
        title_color = [0, 0.6, 0]; % Koyu Yeşil
        durum = 'DOĞRU';
    else
        title_color = [0.8, 0, 0]; % Kırmızı
        durum = 'YANLIŞ';
    end
    
    title_str = sprintf('Gerçek: %s\nTahmin: %s\n[%s]', class_name, pred_class, durum);
    title(title_str, 'Color', title_color, 'FontSize', 11, 'Interpreter', 'none', 'FontWeight', 'bold');
    
    fprintf('Örnek %d | Gerçek: %-12s -> Tahmin: %-12s | Durum: %s\n', i, class_name, pred_class, durum);
end

fprintf('\nTest tamamlandı! Ekrana gelen figürü inceleyebilirsin.\n');