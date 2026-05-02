function hiper4_arayuz
    clc;
    clearvars -except app;
    close all;
%% ================= MODEL YÜKLEME =================
modelPath = 'hiper4_model.mat';

if ~isfile(modelPath)
    uialert(uifigure, ['Model dosyası bulunamadı: ' modelPath], 'Hata');
    return;
end

data = load(modelPath);

if ~isfield(data, 'net')
    uialert(uifigure, 'Model dosyasında "net" bulunamadı.', 'Hata');
    return;
end

net = data.net;

    %% ================= SINIF İSİMLERİ =================
    try
        classNames = string(net.Layers(end).Classes);
    catch
        classNames = [];
    end

    %% ================= ARAYÜZ OLUŞTURMA =================
    fig = uifigure('Name', 'Hiper4 Sınıflandırma Arayüzü', ...
                   'Position', [100 100 1000 600]);

    titleLbl = uilabel(fig, ...
        'Text', 'MATLAB Eğitilmiş Model - Görüntü Sınıflandırma', ...
        'FontSize', 20, ...
        'FontWeight', 'bold', ...
        'Position', [250 550 500 30], ...
        'HorizontalAlignment', 'center');

    % Görsel gösterme alanı
    imgAxes = uiaxes(fig, ...
        'Position', [50 220 350 280]);
    title(imgAxes, 'Seçilen Görüntü');
    imgAxes.XTick = [];
    imgAxes.YTick = [];

    % Sonuç etiketi
    resultLbl = uilabel(fig, ...
        'Text', 'Tahmin: -', ...
        'FontSize', 16, ...
        'FontWeight', 'bold', ...
        'Position', [50 170 400 30]);

    % Güven etiketi
    confLbl = uilabel(fig, ...
        'Text', 'Güven: -', ...
        'FontSize', 15, ...
        'Position', [50 135 400 30]);

    % Dosya yolu etiketi
    fileLbl = uilabel(fig, ...
        'Text', 'Dosya: -', ...
        'FontSize', 12, ...
        'Position', [50 100 900 25]);

    % Skor grafiği alanı
    scoreAxes = uiaxes(fig, ...
        'Position', [470 180 480 320]);
    title(scoreAxes, 'Sınıf Skorları');
    xlabel(scoreAxes, 'Sınıflar');
    ylabel(scoreAxes, 'Olasılık / Skor');

    % Buton: Görsel seç
    btnLoad = uibutton(fig, 'push', ...
        'Text', 'Görsel Seç ve Tahmin Yap', ...
        'FontSize', 15, ...
        'FontWeight', 'bold', ...
        'Position', [50 40 220 40], ...
        'ButtonPushedFcn', @(btn,event) predictImage());

    % Buton: Temizle
    btnClear = uibutton(fig, 'push', ...
        'Text', 'Temizle', ...
        'FontSize', 15, ...
        'Position', [290 40 120 40], ...
        'ButtonPushedFcn', @(btn,event) clearScreen());

    %% ================= TAHMİN FONKSİYONU =================
    function predictImage()
        [file, path] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp;*.tif', ...
                                  'Image Files (*.jpg, *.jpeg, *.png, *.bmp, *.tif)'}, ...
                                  'Bir görüntü seçin');

        if isequal(file, 0)
            return;
        end

        fullPath = fullfile(path, file);
        fileLbl.Text = ['Dosya: ' fullPath];

        try
            % Orijinal resmi oku
            img = imread(fullPath);

            % Gösterim için orijinal resmi çiz
            imshow(img, 'Parent', imgAxes);

            % ===== Eğitimdekiyle aynı preprocessing =====
            imgGray = im2gray(img);
            imgResized = imresize(imgGray, [224 224]);

            % Ağın giriş boyutuna göre ayarla
            inputSize = net.Layers(1).InputSize;

            if numel(inputSize) == 3
                if inputSize(3) == 1
                    imgInput = reshape(imgResized, [224 224 1]);
                elseif inputSize(3) == 3
                    imgInput = cat(3, imgResized, imgResized, imgResized);
                else
                    uialert(fig, 'Beklenmeyen kanal sayısı.', 'Hata');
                    return;
                end
            else
                uialert(fig, 'Model giriş boyutu okunamadı.', 'Hata');
                return;
            end

            % Tahmin
            [predLabel, scores] = classify(net, imgInput);

            % Güven
            [maxScore, idx] = max(scores);
            maxScorePercent = maxScore * 100;

            % Sonuçları yaz
            resultLbl.Text = ['Tahmin: ' char(predLabel)];
            confLbl.Text = sprintf('Güven: %.2f%%', maxScorePercent);

            % Sınıf isimleri
            if isempty(classNames)
                classNamesLocal = strings(1, numel(scores));
                for i = 1:numel(scores)
                    classNamesLocal(i) = "Sınıf " + i;
                end
            else
                classNamesLocal = classNames;
            end

            % Grafik çiz
            cla(scoreAxes);
            bar(scoreAxes, scores);
            scoreAxes.XTick = 1:numel(scores);
            scoreAxes.XTickLabel = classNamesLocal;
            scoreAxes.XTickLabelRotation = 45;
            ylim(scoreAxes, [0 1]);
            grid(scoreAxes, 'on');

            % En yüksek sınıfı başlıkta da göster
            title(scoreAxes, ['Sınıf Skorları | En Yüksek: ' char(predLabel)]);

            % İstersen command window'a da yazdır
            fprintf('\n--- TAHMİN SONUCU ---\n');
            fprintf('Dosya: %s\n', fullPath);
            fprintf('Tahmin edilen sınıf: %s\n', char(predLabel));
            fprintf('Güven: %.2f%%\n', maxScorePercent);

            for i = 1:numel(scores)
                fprintf('%s: %.4f\n', string(classNamesLocal(i)), scores(i));
            end

        catch ME
            uialert(fig, ['Tahmin sırasında hata oluştu: ' ME.message], 'Hata');
            disp(getReport(ME));
        end
    end

    %% ================= TEMİZLEME FONKSİYONU =================
    function clearScreen()
        cla(imgAxes);
        cla(scoreAxes);
        title(imgAxes, 'Seçilen Görüntü');
        title(scoreAxes, 'Sınıf Skorları');
        resultLbl.Text = 'Tahmin: -';
        confLbl.Text = 'Güven: -';
        fileLbl.Text = 'Dosya: -';
    end
end