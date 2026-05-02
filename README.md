# 🧠 Beyin Kanaması Tespit Sistemi

> **Brain Hemorrhage Detection System** — BT görüntülerinden derin öğrenme ile beyin kanaması tespiti

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2022%2B-red)](https://www.mathworks.com/)
---

## 📋 Proje Hakkında

Bu proje, beyin BT (Bilgisayarlı Tomografi) görüntülerinden **beyin kanaması (intrakranial hemoraji)** tespiti yapmak amacıyla geliştirilmiştir. İki farklı model mimarisi kullanılmıştır:

- **ResNet50** — Transfer learning ile Python/TensorFlow ortamında eğitilmiş ikili sınıflandırma modeli (%96.7 doğruluk, F1: 0.966)
- **Özgün CNN (Hiper4)** — MATLAB ortamında Swish aktivasyonlu, VGG-tarzı özgün mimari

---

## ⚠️ Sorumluluk Reddi

> Bu sistem yalnızca **araştırma ve akademik amaçlıdır.**
> Klinik tanı veya tedavi kararı için kullanılamaz.

---

## 📄 Lisans

Bu proje şu anda resmi bir lisans ile yayınlanmamıştır.
Projeyi kopyalamadan veya dağıtmadan önce proje sahipleri ile iletişime geçmeniz gerekmektedir.

### 👥 Proje Sahipleri
- Nilgün Melek Bilgin  
- Gamze Özdemir  
- Çiğdem Kurt  
- Sıla Dertli  
- Hanife Çilingir
- 
---

## 📁 Dosya Yapısı

```
Beyin_Kanaması_Tespiti/
│
├── Resnet50_pretrained_model/
│   ├── best_resnet50_model.keras
│   ├── model_egitim_kodu.ipynb
│   ├── model_egitim_requirements.txt
│   ├── arayuz_requirements.txt
│   └── proje_arayuz_kodları/
│       └── deep_learning_brain_hemorrage/
│           ├── main.py
│           └── models/
│               └── best_resnet50_model.keras
│
├── OzgunMODEL_CNN_Hiper4/
│   ├── Ozgun_CNN_Hiper_4_kod.m
│   ├── hiper4_arayuz.m
│   ├── hiper4_model.mat
│   ├── arayuz_requirements.txt
│   ├── model_requirements.txt
│

```

---

## 🏗️ Model Mimarileri

### ResNet50 (Python / TensorFlow)

```
ResNet50 (ImageNet ağırlıkları, dondurulmuş)
    ↓
GlobalAveragePooling2D
    ↓
Dense(256, ReLU) + Dropout(0.5)
    ↓
Dense(1, Sigmoid)   ← İkili çıkış: Kanama Var / Yok
```

| Parametre | Değer |
|-----------|-------|
| Giriş Boyutu | 224 × 224 × 3 (RGB) |
| Optimizatör | Adam (lr=1e-4) |
| Kayıp Fonksiyonu | Binary Crossentropy |
| Batch Size | 16 |
| Doğruluk | %96.7 |
| F1-Skor | 0.966 |

---

### Özgün CNN — Hiper4 (MATLAB)

```
Input [224×224×1 Gri]
    ↓ Block 1: Conv(3×3, 32) → BN → Swish → MaxPool
    ↓ Block 2: Conv(3×3, 64) × 2 → BN → Swish → MaxPool
    ↓ Block 3: Conv(3×3, 128) → BN → Swish → MaxPool
    ↓ Block 4: Conv(3×3, 256) → BN → Swish
    ↓ GlobalAveragePooling2D
    ↓ FC(128) → ReLU → Dropout(0.6)
    ↓ FC(numClasses) → Softmax
```

| Parametre | Değer |
|-----------|-------|
| Giriş Boyutu | 224 × 224 × 1 (Grayscale) |
| Aktivasyon | Swish |
| Optimizatör | Adam (lr=1e-3) |
| Maksimum Epoch | 70 |
| Batch Size | 32 |
| Veri Dağılımı | %70 Eğitim / %15 Doğrulama / %15 Test |

---

## 🖥️ Python Arayüzü Özellikleri

- BT görüntüsü yükleme (PNG, JPG, BMP, TIF)
- Gerçek zamanlı analiz (arka planda işlem)
- Kanama olasılık çubuğu (renk kodlu: Yeşil / Sarı / Kırmızı)
- Model durumu göstergesi
- Demo modu (TensorFlow olmadan test)
- Klinik uyarı notu

---

## 📊 Veri Seti

Proje, beyin BT görüntülerinden oluşan bir veri seti üzerinde eğitilmiştir.

- **Format:** PNG görüntüler + labels.csv
- **Etiketler:** `hemorrhage` sütunu → `0` (Yok) / `1` (Var)
- **Görüntü Boyutu:** 224 × 224 piksel

> Veri seti gizlilik nedeniyle repoya dahil edilmemiştir.

---

## 📄 Lisans

Bu proje akademik amaçlı geliştirilmiştir. Ticari kullanım için proje sahipleriyle iletişime geçiniz.
