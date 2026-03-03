% src/data_generation/generate_dataset.m
clc; clear; close all;

% Mevcut dosyanın konumunu al (src/data_generation/)
script_path = fileparts(mfilename('fullpath'));

% 2 klasör geriye (Root dizinine) git
% src/data_generation -> src -> Root
project_root = fullfile(script_path, '..', '..'); 

% Klasör yollarını root'a göre tanımla
base_dir = fullfile(project_root, 'data'); 
img_dir = fullfile(base_dir, 'images');
sig_dir = fullfile(base_dir, 'signals');

addpath(genpath(fullfile(project_root, 'src')));

classes = {'Underdamped', 'Overdamped'};

% Klasörleri oluştur (Yoksa oluşturur)
for c = 1:length(classes)
    train_img_path = fullfile(img_dir, 'train', classes{c});
    train_sig_path = fullfile(sig_dir, 'train', classes{c});
    
    if ~exist(train_img_path, 'dir'), mkdir(train_img_path); end
    if ~exist(train_sig_path, 'dir'), mkdir(train_sig_path); end
end

num_samples = 1000;           
t = linspace(0, 5, 2000)';  
u = ones(size(t)); % İLERİDE DUAL-INPUT İÇİN GEREKLİ (Step Input)

target_img_size = [228, 448]; 
fig = figure('Visible', 'off', 'Color', 'k'); 

for c = 1:length(classes)
    class_name = classes{c};
    fprintf('%s verileri üretiliyor (Klasör: %s)...\n', class_name, base_dir);
    
    for i = 1:num_samples
        K = 0.5 + rand() * 4.5;
        
        if strcmp(class_name, 'Underdamped')
            zeta = 0.05 + rand() * 0.85; 
            wn = 2 + rand() * 8;         
        else
            zeta = 1.1 + rand() * 0.9;   
            wn = 2 + rand() * 8;
        end
        
        sys = tf([K * wn^2], [1, 2*zeta*wn, wn^2]);
        [y, t_out] = step(sys, t);
        
        % DATA KAYDI (Root/data/signals/train/...)
        sig_filename = fullfile(sig_dir, 'train', class_name, sprintf('sample_%03d.mat', i));
        % u (input) sinyalini de ekledik, ilerde hoca sorarsa 'girişi de tutuyorum' dersin
        save(sig_filename, 'y', 'u', 't', 'zeta', 'wn', 'K');
        
        % GÖRSELLEŞTİRME (Hard Mode)
        plot(t, y, 'Color', 'w', 'LineWidth', 2);
        xlim([0, 5]); ylim([0, 7]); 
        axis off; 
        
        % RESİM KAYDI (Root/data/images/train/...)
        img_filename = fullfile(img_dir, 'train', class_name, sprintf('sample_%03d.png', i));
        process_plot_to_image(gca, img_filename, target_img_size);
        
        clf; 
    end
end
close(fig);
disp('Veri üretimi Root/data klasörüne başarıyla tamamlandı!');