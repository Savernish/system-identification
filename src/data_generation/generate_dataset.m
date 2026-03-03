% src/data_generation/generate_dataset.m
clc; clear; close all;

addpath(genpath(fullfile(pwd, 'src')));

base_dir = fullfile(pwd, 'data'); 
img_dir = fullfile(base_dir, 'images');
sig_dir = fullfile(base_dir, 'signals');

classes = {'Underdamped', 'Overdamped'};

for c = 1:length(classes)
    if ~exist(fullfile(img_dir, 'train', classes{c}), 'dir')
        mkdir(fullfile(img_dir, 'train', classes{c}));
    end
    if ~exist(fullfile(sig_dir, 'train', classes{c}), 'dir')
        mkdir(fullfile(sig_dir, 'train', classes{c}));
    end
end

num_samples = 1000;           
t = linspace(0, 5, 2000)';  
target_img_size = [228, 448]; % Sadece boyut var, padding yok

fig = figure('Visible', 'off', 'Color', 'k'); 

for c = 1:length(classes)
    class_name = classes{c};
    fprintf('%s verileri üretiliyor...\n', class_name);
    
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
        
        sig_filename = fullfile(sig_dir, 'train', class_name, sprintf('sample_%03d.mat', i));
        save(sig_filename, 'y', 't', 'zeta', 'wn', 'K');
        
        % Çizgi
        plot(t, y, 'Color', 'w', 'LineWidth', 2);
        
        xlim([0, 5]); ylim([0, 7]); 
        
        % BÜTÜN EKSENLERİ, ÇERÇEVELERİ VE GRİDLERİ YOK ET (HARD MODE)
        axis off; 
        
        img_filename = fullfile(img_dir, 'train', class_name, sprintf('sample_%03d.png', i));
        
        % Artık fonksiyona padding yollamıyoruz
        process_plot_to_image(gca, img_filename, target_img_size);
        
        clf; 
    end
end

close(fig);
disp('Hard Mode veri üretimi başarıyla tamamlandı!');