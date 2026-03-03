% src/training/train_model.m
clc; clear; close all;

script_dir   = fileparts(mfilename('fullpath'));
project_root = fullfile(script_dir, '..', '..');

base_dir = fullfile(project_root, 'data');
img_dir  = fullfile(base_dir, 'images', 'train');
sig_dir  = fullfile(base_dir, 'signals', 'train');

addpath(genpath(fullfile(project_root, 'src')));

classes = {'Underdamped','Overdamped'};
num_classes = numel(classes);
samples_per_class = 1000;
total_samples = num_classes * samples_per_class;

fprintf('Veriler yükleniyor...\n');

Col_Img = cell(total_samples,1);
Col_Sig = cell(total_samples,1);
Y_str   = strings(total_samples,1);

idx = 1;

for c = 1:num_classes
    class_name = classes{c};
    
    for i = 1:samples_per_class
        
        % ---- IMAGE ----
        img_path = fullfile(img_dir,class_name,...
            sprintf('sample_%03d.png',i));
        
        img = imread(img_path);
        
        if ismatrix(img)
            img = reshape(img,size(img,1),size(img,2),1);
        end
        
        img = im2single(img);   % CNN için önerilir
        
        Col_Img{idx} = img;
        
        
        % ---- SIGNAL ----
        sig_path = fullfile(sig_dir,class_name,...
            sprintf('sample_%03d.mat',i));
        
        data = load(sig_path);
        
        sig = double(data.y(:))';   % 1x2000 zorla
        
        Col_Sig{idx} = sig;
        
        
        % ---- LABEL ----
        Y_str(idx) = class_name;
        
        idx = idx + 1;
    end
end

Y = categorical(Y_str);

fprintf('Ağ oluşturuluyor...\n');
lgraph = build_fusion_net();


%% =============================
%   DATASTORE PIPELINE
% ==============================

fprintf('Datastore kuruluyor...\n');

dsImg = arrayDatastore(Col_Img,'IterationDimension',1);
dsSig = arrayDatastore(Col_Sig,'IterationDimension',1);
dsY   = arrayDatastore(Y,'IterationDimension',1);

% Hücreyi aç (KRİTİK)
dsImg = transform(dsImg,@(x)x{1});
dsSig = transform(dsSig,@(x)x{1});

% Giriş sırasına göre combine
inputNames = lgraph.InputNames;

if strcmp(inputNames{1},'input_2D')
    dsTrain = combine(dsImg,dsSig,dsY);
    fprintf('[Resim | Sinyal | Label]\n');
else
    dsTrain = combine(dsSig,dsImg,dsY);
    fprintf('[Sinyal | Resim | Label]\n');
end


%% =============================
%   TRAINING OPTIONS
% ==============================

options = trainingOptions('adam', ...
    'MaxEpochs',20,...
    'MiniBatchSize',4,...
    'Shuffle','every-epoch',...
    'InitialLearnRate',1e-3,...
    'Verbose',false,...
    'Plots','training-progress');


fprintf('Eğitim başlıyor...\n');

[net,info] = trainNetwork(dsTrain,lgraph,options);


%% =============================
%   SAVE MODEL
% ==============================

results_dir = fullfile(project_root,'results');
if ~exist(results_dir,'dir')
    mkdir(results_dir);
end

save(fullfile(results_dir,'multimodal_poc_net.mat'),'net','info');

fprintf('Model kaydedildi.\n');