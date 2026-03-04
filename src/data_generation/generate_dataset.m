% src/data_generation/generate_dataset.m
clc; clear; close all;

% Proje kök dizinini bul
script_path = fileparts(mfilename('fullpath'));
project_root = fullfile(script_path, '..', '..'); 
addpath(genpath(fullfile(project_root, 'src')));

% 6 Sınıflı Grand Model Kategorileri
classes = {'Class1_FOPTD', 'Class2_SOPTD', 'Class3_NMP', ...
           'Class4_Underdamped', 'Class5_HighOrder', 'Class6_Integrator'};
splits = {'train', 'val', 'test'};

base_dir = fullfile(project_root, 'data');
img_dir = fullfile(base_dir, 'images');
sig_dir = fullfile(base_dir, 'signals');

% Klasör Ağacını Otomatik Kur
for s = 1:length(splits)
    for c = 1:length(classes)
        if ~exist(fullfile(img_dir, splits{s}, classes{c}), 'dir')
            mkdir(fullfile(img_dir, splits{s}, classes{c}));
        end
        if ~exist(fullfile(sig_dir, splits{s}, classes{c}), 'dir')
            mkdir(fullfile(sig_dir, splits{s}, classes{c}));
        end
    end
end

% Veri Dağılımı (Sınıf Başına 1000 Veri -> Toplam 6000)
split_sizes = struct('train', 700, 'val', 150, 'test', 150);

% Simülasyon Zamanı (Sistemler yavaş olabileceği için 30 saniyeye çıkardık)
t = linspace(0, 30, 2000)';  
u = ones(size(t)); % Dual-Input için birim basamak (Step) giriş sinyali

target_img_size = [228, 448]; 
fig = figure('Visible', 'off', 'Color', 'k'); 

for s = 1:length(splits)
    split_name = splits{s};
    num_samples = split_sizes.(split_name);
    
    fprintf('\n==================================================\n');
    fprintf('--- %s SETİ ÜRETİLİYOR (%d Örnek/Sınıf) ---\n', upper(split_name), num_samples);
    fprintf('==================================================\n');
    
    for c = 1:length(classes)
        class_name = classes{c};
        fprintf('%-20s üretiliyor...\n', class_name);
        
        for i = 1:num_samples
            % Ortak Parametreler
            K = 0.5 + rand() * 4.5;    % Kazanç [0.5, 5.0]
            L = 0.1 + rand() * 2.9;    % Ölü Zaman (Dead-Time) [0.1, 3.0] saniye
            
            % Sınıflara Özel Transfer Fonksiyonu (G(s)) İnşası
            switch class_name
                case 'Class1_FOPTD'
                    T = 0.5 + rand() * 4.5;
                    sys = tf(K, [T, 1]);
                    
                case 'Class2_SOPTD'
                    T1 = 0.5 + rand() * 4.5;
                    T2 = 0.5 + rand() * 4.5;
                    sys = tf(K, conv([T1, 1], [T2, 1]));
                    
                case 'Class3_NMP'
                    % Undershoot yaratan Sağ Yarı Düzlem Sıfırı (Tz)
                    T1 = 0.5 + rand() * 3.5;
                    T2 = 0.5 + rand() * 3.5;
                    Tz = 0.5 + rand() * 2.5; 
                    sys = tf([-K*Tz, K], conv([T1, 1], [T2, 1]));
                    
                case 'Class4_Underdamped'
                    zeta = 0.1 + rand() * 0.7; 
                    wn = 0.5 + rand() * 2.5;         
                    sys = tf(K * wn^2, [1, 2*zeta*wn, wn^2]);
                    
                case 'Class5_HighOrder'
                    % 4. Dereceden atıl/hantal sistem
                    T1 = 0.5 + rand() * 2; T2 = 0.5 + rand() * 2;
                    T3 = 0.5 + rand() * 2; T4 = 0.5 + rand() * 2;
                    p1 = conv([T1, 1], [T2, 1]);
                    p2 = conv([T3, 1], [T4, 1]);
                    sys = tf(K, conv(p1, p2));
                    
                case 'Class6_Integrator'
                    % Entegratörlü (1/s) yapı, doyum yapmaz
                    T = 0.5 + rand() * 4.5;
                    sys = tf(K, [T, 1, 0]);
            end
            
            % Gecikmeyi (Dead-Time) Sisteme Ekle
            sys.InputDelay = L;
            
            % Basamak Yanıtını Al (Temiz Sinyal)
            [y_clean, ~] = step(sys, t);
            
            % =========================================================
            % +++ 1. YENİLİK: %1 GAUSSIAN NOISE (GÜRÜLTÜ) EKLEME +++
            % =========================================================
            % Sinyalin genlik aralığının (max-min) %1'i kadar gürültü ekliyoruz.
            % Böylece K=5 olan sistemle K=0.5 olan sistem orantılı gürültü alır.
            y_range = max(y_clean) - min(y_clean);
            if y_range == 0; y_range = 1; end % Sıfıra bölme/çarpma koruması
            
            noise = 0.01 * y_range * randn(size(y_clean)); 
            y = y_clean + noise; % Artık y sinyalimiz hafif titrek/gerçekçi
            
            % 1. SİNYALİ KAYDET (.mat) -> Kaydederken Gürültülü Halini Kaydediyoruz!
            sig_filename = fullfile(sig_dir, split_name, class_name, sprintf('sample_%03d.mat', i));
            save(sig_filename, 'y', 'u', 't', 'K', 'L');
            
            % =========================================================
            % +++ 2. YENİLİK: RANDOM LINEWIDTH (ÇİZGİ KALINLIĞI) +++
            % =========================================================
            % Çizgi kalınlığı 1.0 ile 3.0 arasında rastgele değişecek
            rand_lw = 1.0 + rand() * 2.0; 
            
            % 2. GÖRÜNTÜYÜ OLUŞTUR VE KAYDET (.png)
            plot(t, y, 'Color', 'w', 'LineWidth', rand_lw);
            
            % Dinamik Eksene Oturtma (Önceki Kırpılma Hatası Çözülmüş Hali)
            xlim([0, max(t)]); 
            
            y_min = min(y);
            y_max = max(y);
            y_margin = (y_max - y_min) * 0.1; % Alt ve üste %10 boşluk
            if y_margin == 0; y_margin = 0.1; end 
            
            ylim([y_min - y_margin, y_max + y_margin]);
            axis off; % Hard Mode: Sadece titrek, rastgele kalınlıkta beyaz çizgi
            
            img_filename = fullfile(img_dir, split_name, class_name, sprintf('sample_%03d.png', i));
            process_plot_to_image(gca, img_filename, target_img_size);
            
            clf;
        end
    end
end

close(fig);
fprintf('\nGRAND MODEL VERİ SETİ (6000 Örnek) BAŞARIYLA ÜRETİLDİ!\n');