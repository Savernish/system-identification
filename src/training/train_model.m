% src/training/train_model.m
clc; clear; close all;

% Proje kök dizinini bul
script_path = fileparts(mfilename('fullpath'));
project_root = fullfile(script_path, '..', '..');
addpath(genpath(fullfile(project_root, 'src')));

%% 1. Veri Yollarını Belirle (Train ve Val)
train_img_dir = fullfile(project_root, 'data', 'images', 'train');
train_sig_dir = fullfile(project_root, 'data', 'signals', 'train');

val_img_dir = fullfile(project_root, 'data', 'images', 'val');
val_sig_dir = fullfile(project_root, 'data', 'signals', 'val');

%% =========================================================
%% 2. TRAIN DATASTORE KURULUMU
%% =========================================================
fprintf('Eğitim (Train) verileri belleğe alınıyor...\n');
imdsTrain = imageDatastore(train_img_dir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
labelsTrain = imdsTrain.Labels;

Col_Img_Train = cell(length(imdsTrain.Files), 1);
Col_Sig_Train = cell(length(imdsTrain.Files), 1);

for i = 1:length(imdsTrain.Files)
    % Resmi Oku
    img = imread(imdsTrain.Files{i});
    if ismatrix(img), img = reshape(img, size(img,1), size(img,2), 1); end
    Col_Img_Train{i} = im2single(img);
    
    % İlgili Sinyali Oku
    [~, name, ~] = fileparts(imdsTrain.Files{i});
    class_name = char(labelsTrain(i));
    sig_file = fullfile(train_sig_dir, class_name, [name, '.mat']);
    data = load(sig_file);
    Col_Sig_Train{i} = double(data.y(:))'; % 1x2000
end

% Train için Combine işlemi (Hücreleri Açıyoruz)
dsImgTrain = transform(arrayDatastore(Col_Img_Train, 'IterationDimension', 1), @(x) x{1});
dsSigTrain = transform(arrayDatastore(Col_Sig_Train, 'IterationDimension', 1), @(x) x{1});
dsYTrain = arrayDatastore(labelsTrain);
dsTrain = combine(dsImgTrain, dsSigTrain, dsYTrain); % Giriş sırası: [Resim, Sinyal, Etiket]

%% =========================================================
%% 3. VALIDATION DATASTORE KURULUMU (YENİ EKLENDİ)
%% =========================================================
fprintf('Doğrulama (Validation) verileri belleğe alınıyor...\n');
imdsVal = imageDatastore(val_img_dir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
labelsVal = imdsVal.Labels;

Col_Img_Val = cell(length(imdsVal.Files), 1);
Col_Sig_Val = cell(length(imdsVal.Files), 1);

for i = 1:length(imdsVal.Files)
    % Resmi Oku
    img = imread(imdsVal.Files{i});
    if ismatrix(img), img = reshape(img, size(img,1), size(img,2), 1); end
    Col_Img_Val{i} = im2single(img);
    
    % İlgili Sinyali Oku
    [~, name, ~] = fileparts(imdsVal.Files{i});
    class_name = char(labelsVal(i));
    sig_file = fullfile(val_sig_dir, class_name, [name, '.mat']);
    data = load(sig_file);
    Col_Sig_Val{i} = double(data.y(:))'; 
end

% Val için Combine işlemi
dsImgVal = transform(arrayDatastore(Col_Img_Val, 'IterationDimension', 1), @(x) x{1});
dsSigVal = transform(arrayDatastore(Col_Sig_Val, 'IterationDimension', 1), @(x) x{1});
dsYVal = arrayDatastore(labelsVal);
dsVal = combine(dsImgVal, dsSigVal, dsYVal);

%% =========================================================
%% 4. AĞI YÜKLE VE EĞİTİM SEÇENEKLERİNİ AYARLA
%% =========================================================
lgraph = build_fusion_net(); % Senin o efsanevi ağ mimarin

miniBatchSize = 32;
% Validation Frequency: Her epoch bitiminde validation yapsın diye hesaplıyoruz
valFreq = floor(length(imdsTrain.Files) / miniBatchSize); 

options = trainingOptions('sgdm', ...
    'MiniBatchSize', miniBatchSize, ...
    'MaxEpochs', 40, ...                  % Epoch'u biraz artırdık çünkü Early Stopping var
    'InitialLearnRate', 1e-3, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', dsVal, ...          % <--- İŞTE ŞOV BURADA BAŞLIYOR
    'ValidationFrequency', valFreq, ...   % <--- Her epoch'ta bir test et
    'ValidationPatience', 4, ...          % <--- 4 Epoch boyunca Val Loss düşmezse eğitimi KES! (Overfit Kalkanı)
    'Plots', 'training-progress', ...
    'Verbose', false);

%% =========================================================
%% 5. EĞİTİMİ BAŞLAT VE KAYDET
%% =========================================================
fprintf('\nEĞİTİM BAŞLIYOR! Lütfen ekrana gelecek olan eğitim grafiğini izleyin...\n');
net = trainNetwork(dsTrain, lgraph, options);

% Eğitilen modeli kaydet
save_dir = fullfile(project_root, 'results');
if ~exist(save_dir, 'dir'), mkdir(save_dir); end
save(fullfile(save_dir, 'multimodal_poc_net.mat'), 'net');

fprintf('Model başarıyla eğitildi ve "results/multimodal_poc_net.mat" olarak kaydedildi!\n');