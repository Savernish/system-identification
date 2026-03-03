% src/data_generation/process_plot_to_image.m
function process_plot_to_image(ax, filename, target_size)
    % Grafiği yakalar, binarize eder ve doğrudan istenen boyuta getirir. (Kenarlıksız)

    temp_file = 'temp_capture.png';
    exportgraphics(ax, temp_file, 'Resolution', 150, 'BackgroundColor', 'k');
    
    img_rgb = imread(temp_file);
    delete(temp_file);
    
    if size(img_rgb, 3) == 3
        img_gray = rgb2gray(img_rgb);
    else
        img_gray = img_rgb;
    end
    
    % Sabit Binarizasyon
    img_bin = img_gray > 100;
    
    % Doğrudan hedef boyuta yeniden boyutlandır
    final_img = imresize(img_bin, target_size, 'nearest');
    
    % Kaydet
    imwrite(final_img, filename);
end