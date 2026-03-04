% src/models/build_fusion_net.m
function lgraph = build_fusion_net()
    % Multimodal 6-Class Fusion Network (Dual-Input 1D + 2D CNN)
    
    lgraph = layerGraph();
    
    % =====================================================================
    % 1. KOL: GÖRÜNTÜ İŞLEME (2D CNN - Morfoloji)
    % =====================================================================
    layers_2D = [
        imageInputLayer([228 448 1], 'Name', 'input_2D', 'Normalization', 'none')
        
        convolution2dLayer(5, 16, 'Padding', 'same', 'Name', 'conv_2d_1')
        batchNormalizationLayer('Name', 'bn_2d_1')
        reluLayer('Name', 'relu_2d_1')
        maxPooling2dLayer(4, 'Stride', 4, 'Name', 'pool_2d_1') 
        
        convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv_2d_2')
        batchNormalizationLayer('Name', 'bn_2d_2')
        reluLayer('Name', 'relu_2d_2')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool_2d_2') 
        
        % Kapasiteyi artırdık: 6 sınıf ve farklı çizgi kalınlıkları için
        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv_2d_3')
        batchNormalizationLayer('Name', 'bn_2d_3')
        reluLayer('Name', 'relu_2d_3')
        
        globalAveragePooling2dLayer('Name', 'gap_2d') 
        fullyConnectedLayer(128, 'Name', 'fc_2D_feat') 
        flattenLayer('Name', 'flatten_2D') 
    ];
    lgraph = addLayers(lgraph, layers_2D);
    
    % =====================================================================
    % 2. KOL: ZAMAN SERİSİ İŞLEME (1D CNN - Temporal & Kazanç/Gecikme)
    % =====================================================================
    layers_1D = [
        % DİKKAT: Artık giriş boyutu 2! (u ve y kanalları)
        sequenceInputLayer(2, 'Name', 'input_1D', 'MinLength', 2000)
        
        convolution1dLayer(10, 16, 'Padding', 'same', 'Name', 'conv_1d_1')
        batchNormalizationLayer('Name', 'bn_1d_1')
        reluLayer('Name', 'relu_1d_1')
        maxPooling1dLayer(4, 'Stride', 4, 'Name', 'pool_1d_1')
        
        convolution1dLayer(5, 32, 'Padding', 'same', 'Name', 'conv_1d_2')
        batchNormalizationLayer('Name', 'bn_1d_2')
        reluLayer('Name', 'relu_1d_2')
        maxPooling1dLayer(2, 'Stride', 2, 'Name', 'pool_1d_2') 
        
        convolution1dLayer(3, 64, 'Padding', 'same', 'Name', 'conv_1d_3')
        batchNormalizationLayer('Name', 'bn_1d_3')
        reluLayer('Name', 'relu_1d_3')
        
        globalAveragePooling1dLayer('Name', 'gap_1d') 
        fullyConnectedLayer(128, 'Name', 'fc_1D_feat') 
        flattenLayer('Name', 'flatten_1D') 
    ];
    lgraph = addLayers(lgraph, layers_1D);
    
    % =====================================================================
    % 3. BİRLEŞTİRME (LATE FUSION) VE 6 SINIFLI ÇIKIŞ KAFASI
    % =====================================================================
    layers_fusion = [
        concatenationLayer(1, 2, 'Name', 'concat')
        
        % Derin Sınıflandırıcı (6 Karmaşık Sınıf İçin)
        fullyConnectedLayer(128, 'Name', 'fc_fusion_1')
        reluLayer('Name', 'relu_fusion_1')
        dropoutLayer(0.4, "Name", "dp_fusion_1") % Gürültüye karşı Dropout'u azıcık artırdık
        
        fullyConnectedLayer(64, 'Name', 'fc_fusion_2')
        reluLayer('Name', 'relu_fusion_2')
        dropoutLayer(0.3, "Name", "dp_fusion_2")
        
        % DİKKAT: Çıkış artık 6 Sınıf!
        fullyConnectedLayer(6, 'Name', 'fc_output') 
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'classoutput')
    ];
    lgraph = addLayers(lgraph, layers_fusion);
    
    % =====================================================================
    % 4. KOLLARI BİRBİRİNE BAĞLA
    % =====================================================================
    lgraph = connectLayers(lgraph, 'flatten_2D', 'concat/in1');
    lgraph = connectLayers(lgraph, 'flatten_1D', 'concat/in2');
    
end