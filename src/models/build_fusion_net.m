% src/models/build_fusion_net.m
function lgraph = build_fusion_net()
    % Multimodal (1D Zaman Serisi + 2D Görüntü) DAG Ağ Mimarisi
    
    lgraph = layerGraph();

    % =====================================================================
    % 1. KOL: GÖRÜNTÜ İŞLEME (2D CNN)
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
        globalAveragePooling2dLayer('Name', 'gap_2d')
        
        fullyConnectedLayer(64, 'Name', 'fc_2D_feat') 
        flattenLayer('Name', 'flatten_2D') % <--- FORMAT DÜZELTİCİ (YENİ)
    ];
    lgraph = addLayers(lgraph, layers_2D);

    % =====================================================================
    % 2. KOL: ZAMAN SERİSİ İŞLEME (1D CNN)
    % =====================================================================
    layers_1D = [
        sequenceInputLayer(1, 'Name', 'input_1D', 'MinLength', 2000)
        
        convolution1dLayer(10, 16, 'Padding', 'same', 'Name', 'conv_1d_1')
        batchNormalizationLayer('Name', 'bn_1d_1')
        reluLayer('Name', 'relu_1d_1')
        maxPooling1dLayer(4, 'Stride', 4, 'Name', 'pool_1d_1')
        
        convolution1dLayer(5, 32, 'Padding', 'same', 'Name', 'conv_1d_2')
        batchNormalizationLayer('Name', 'bn_1d_2')
        reluLayer('Name', 'relu_1d_2')
        globalAveragePooling1dLayer('Name', 'gap_1d')
        
        fullyConnectedLayer(64, 'Name', 'fc_1D_feat') 
        flattenLayer('Name', 'flatten_1D') % <--- FORMAT DÜZELTİCİ (YENİ)
    ];
    lgraph = addLayers(lgraph, layers_1D);

    % =====================================================================
    % 3. BİRLEŞTİRME (LATE FUSION) VE SINIFLANDIRICI KAFASI
    % =====================================================================
    layers_fusion = [
        concatenationLayer(1, 2, 'Name', 'concat')
        
        fullyConnectedLayer(64, 'Name', 'fc_fusion_1')
        reluLayer('Name', 'relu_fusion1')
        dropoutLayer(0.3, "Name", "dp_fusion_1")
        reluLayer("Name", "relu_fusion2")
        fullyConnectedLayer(2, 'Name', 'fc_output') % 2 Sınıf
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'classoutput')
    ];
    lgraph = addLayers(lgraph, layers_fusion);

    % =====================================================================
    % 4. KOLLARI BİRBİRİNE BAĞLA
    % =====================================================================
    % Artık FC katmanlarını değil, Flatten katmanlarını birleştiriciye bağlıyoruz
    lgraph = connectLayers(lgraph, 'flatten_2D', 'concat/in1');
    lgraph = connectLayers(lgraph, 'flatten_1D', 'concat/in2');
    
end