rng(8); % Sonuçların tekrarlanabilirliği için

% Çıktı klasörü oluşturma
outDir = 'OzgunMODEL_CNN_Hiper4';
if ~exist(outDir,'dir')
    mkdir(outDir);
end

%% ================= VERİ YÜKLEME =================
imgDir = 'C:\Users\user\Desktop\DATA\DATA'; 
imds = imageDatastore(imgDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Görüntüleri 224x224 boyutuna ve Gri Seviyeye (Grayscale) dönüştürme
imds.ReadFcn = @(x) imresize(im2gray(imread(x)), [224 224]);
numClasses = numel(categories(imds.Labels));

%% ================= VERİ SETİ AYIRMA =================
% %70 Eğitim, %15 Doğrulama (Validation), %15 Test
[imdsTrain, imdsTemp] = splitEachLabel(imds, 0.70, 'randomized');
[imdsVal, imdsTest]   = splitEachLabel(imdsTemp, 0.5, 'randomized');

%% ================= VERİ ARTIRMA (AUGMENTATION) =================
% Modelin ezberlemesini önlemek için özgün varyasyonlar ekliyoruz
augmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandRotation', [-15 15], ...
    'RandXTranslation', [-10 10], ...
    'RandYTranslation', [-10 10], ...
    'RandScale', [0.9 1.1]);

augTrain = augmentedImageDatastore([224 224 1], imdsTrain, 'DataAugmentation', augmenter);
augVal   = augmentedImageDatastore([224 224 1], imdsVal);
augTest  = augmentedImageDatastore([224 224 1], imdsTest);

%% ================= ÖZGÜN CNN MİMARİSİ =================
% Bu mimari, VGG-stil derinleşme ile modern Swish aktivasyonunu birleştirir
layers = [
    imageInputLayer([224 224 1], 'Normalization', 'rescale-zero-one', 'Name', 'input')

    % Block 1: Temel Özellikler
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    swishLayer('Name', 'swish1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1') % 112x112

    % Block 2: Derin Doku Analizi
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    swishLayer('Name', 'swish2')
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2_extra') % Özgünlük için ekstra katman
    batchNormalizationLayer('Name', 'bn2_extra')
    swishLayer('Name', 'swish2_extra')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2') % 56x56

    % Block 3: Yüksek Seviyeli Özellikler
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    swishLayer('Name', 'swish3')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool3') % 28x28

    % Block 4: Final Semantik Katman
    convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'conv4')
    batchNormalizationLayer('Name', 'bn4')
    swishLayer('Name', 'swish4')

    % Classifier (Sınıflandırıcı) Bölümü
    globalAveragePooling2dLayer('Name', 'gap') % Flatten yerine GAP (Overfitting önleyici)

    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'relu_final')
    dropoutLayer(0.6, 'Name', 'drop_final') % %40 seyreltme

    fullyConnectedLayer(numClasses, 'Name', 'fc_out')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

%% ================= EĞİTİM SEÇENEKLERİ =================
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 10, ...
    'L2Regularization', 5e-4, ...
    'MaxEpochs', 70, ...
    'MiniBatchSize', 32, ...
    'ValidationData', augVal, ...
    'ValidationFrequency', 10, ...
    'ValidationPatience', 12, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', false);

%% ================= MODEL EĞİTİMİ =================
fprintf('Eğitim başlatılıyor...\n');
[net, info] = trainNetwork(augTrain, layers, options);

%% ================= TEST VE SONUÇLAR =================
fprintf('Test ediliyor...\n');
[pred, scores] = classify(net, augTest);
trueLabels = imdsTest.Labels;
accuracy = mean(pred == trueLabels);

%% ================= GÖRSELLEŞTİRME VE KAYIT =================
% Karmaşıklık Matrisi
fig = figure;
cmChart = confusionchart(trueLabels, pred);
cmChart.Title = ['hiper4 - Doğruluk: %', num2str(accuracy*100, '%.2f')];
saveas(fig, fullfile(outDir, 'final_confusion_matrix.png'));

% Modeli Kaydetme
save(fullfile(outDir, 'hiper4_model.mat'), 'net', 'info');

% Metriklerin Hesaplanması (Precision, Recall, F1)
cm = confusionmat(trueLabels, pred);
precision = diag(cm) ./ (sum(cm, 1)' + eps);
recall    = diag(cm) ./ (sum(cm, 2) + eps);
f1        = 2 * (precision .* recall) ./ (precision + recall + eps);

% Raporu Yazdırma
fprintf('\n--- FİNAL SONUÇLARI ---\n');
fprintf('Genel Doğruluk: %.2f%%\n', accuracy * 100);
fprintf('Macro F1-Skor:  %.4f\n', mean(f1));

%% ================= REQUIREMENTS.TXT OLUŞTURMA (DOĞRU) =================

reqFile = fullfile(outDir, 'requirements.txt');
fid = fopen(reqFile, 'w');

fprintf(fid, 'MATLAB Requirements Report\n');
fprintf(fid, '==========================\n\n');

fprintf(fid, 'MATLAB Version: %s\n\n', version);

% 🔥 Projede gerçekten kullanılan dosya ve toolbox'ları bul
[files, products] = matlab.codetools.requiredFilesAndProducts(mfilename('fullpath'));

fprintf(fid, 'Required Toolboxes / Products:\n');
for i = 1:length(products)
    fprintf(fid, '- %s (Version %s)\n', products(i).Name, products(i).Version);
end

fprintf(fid, '\nRequired Files:\n');
for i = 1:length(files)
    fprintf(fid, '- %s\n', files{i});
end

fclose(fid);

fprintf('requirements.txt sadece kullanılan bağımlılıklarla oluşturuldu.\n');
%% ================= FEATURE CSV OLUŞTURMA =================
fprintf('Feature çıkarma işlemi başlatılıyor...\n');

% Test verisi için feature çıkarma (GAP katmanı)
testFeatures = activations(net, augTest, 'gap', 'OutputAs', 'rows');

% Test label'ları
testLabels = imdsTest.Labels;

% Test feature tablosu
T_test = array2table(testFeatures);
T_test.Label = testLabels;

% Test CSV kaydı
testCSV = fullfile(outDir, 'features_test.csv');
writetable(T_test, testCSV);

fprintf('Test feature CSV kaydedildi: %s\n', testCSV);

%% ================= TRAIN FEATURE CSV =================
trainFeatures = activations(net, augTrain, 'gap', 'OutputAs', 'rows');
trainLabels = imdsTrain.Labels;

T_train = array2table(trainFeatures);
T_train.Label = trainLabels;

trainCSV = fullfile(outDir, 'features_train.csv');
writetable(T_train, trainCSV);

fprintf('Train feature CSV kaydedildi: %s\n', trainCSV);

%% ================= VAL FEATURE CSV =================
valFeatures = activations(net, augVal, 'gap', 'OutputAs', 'rows');
valLabels = imdsVal.Labels;

T_val = array2table(valFeatures);
T_val.Label = valLabels;

valCSV = fullfile(outDir, 'features_val.csv');
writetable(T_val, valCSV);

fprintf('Validation feature CSV kaydedildi: %s\n', valCSV);

%% ================= TEK BİR MASTER CSV =================
allFeatures = [trainFeatures; valFeatures; testFeatures];
allLabels = [trainLabels; valLabels; testLabels];

T_all = array2table(allFeatures);
T_all.Label = allLabels;

allCSV = fullfile(outDir, 'features_all.csv');
writetable(T_all, allCSV);

fprintf('Tüm dataset feature CSV kaydedildi: %s\n', allCSV);
