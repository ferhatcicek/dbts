# 🧠 Deformasyonel Brakisefali Tespit Sistemi

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.x-orange?logo=gradio)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey?logo=windows)](#)

Kafa fotoğraflarından **Sefalik İndeks (SI)** ve **Kranyal Kasa Asimetri İndeksi (KKAI/CVAI)** hesaplayarak Deformasyonel Brakisefali şiddetini değerlendiren yapay zeka destekli klinik karar destek uygulaması.

---

## 📋 Özellikler

| Özellik | Detay |
|---|---|
| **Birincil Analiz** | Üstten görüntüden SI hesabı (GrabCut + rembg AI) |
| **İkincil Analiz** | Önden görüntüde MediaPipe yüz asimetrisi analizi |
| **Ölçümler** | Sefalik İndeks, CVAI, Simetri Skoru, Dairesellik |
| **Sınıflandırma** | Normal / Hafif / Orta / Ağır Brakisefali |
| **Görselleştirme** | Annotasyonlu kafa görüntüsü, ölçüm eksenlerı, SI göstergesi |
| **Raporlama** | PDF + HTML + metin raporu |
| **Arayüz** | Gradio web uygulaması (tarayıcı tabanlı) |

---

## 🏥 Tıbbi Arka Plan

**Deformasyonel Brakisefali** (Pozisyonel Brakisefali), kafanın arka kısmının dışsal baskı nedeniyle yassılaşmasıyla oluşur. Kraniosinostosis'ten (sütür füzyonu) ayırt edilmesi kritik önem taşır.

### Sefalik İndeks Referans Değerleri

| SI Aralığı | Sınıflandırma |
|---|---|
| SI < 75% | Dolikosefali (uzun/dar kafa) |
| 75–84% | **Normal** |
| 85–89% | Hafif Brakisefali |
| 90–95% | Orta Şiddetli Brakisefali |
| >95% | **Ağır Brakisefali** |

---

## ⚙️ Kurulum

### Gereksinimler
- Python 3.9 veya üzeri
- Windows 10/11

### Hızlı Kurulum (Windows)

```batch
setup_venv.bat
```

### Manuel Kurulum

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 Kullanım

### Web Arayüzü (Önerilen)

```batch
run.bat
```

veya:

```bash
.\venv\Scripts\activate
python main.py
```

Tarayıcıda `http://localhost:7860` adresine gidin.

### Komut Satırı

```bash
python main.py --cli fotograf.jpg --view top
python main.py --cli fotograf.jpg --view front
```

### Kütüphane Testi

```bash
python main.py --test
```

---

## 📷 Fotoğraf Çekim Rehberi

### Üstten Görünüm (Birincil — Brakisefali Analizi)
1. Bebeği **sırt üstü** yatırın veya dik oturtun
2. Kamerayı kafanın tam **üzerinde** tutun (kuşbakışı)  
3. Kafanın tamamı **fotoğraf karesinde** olsun
4. **İyi aydınlatma**: gölge olmadan, düzgün arka plan
5. Mümkünse saçlar toplanmış/kısa olsun

### Önden Görünüm
- Bebek kameraya düz bakmalı, yüz hafif meyilli olmamalı

---

## 🗂️ Proje Yapısı

```
kafatasi/
├── setup_venv.bat          # Otomatik kurulum (Windows)
├── run.bat                 # Uygulamayı başlat
├── requirements.txt        # Python bağımlılıkları
├── main.py                 # Giriş noktası
├── app.py                  # Gradio web arayüzü
├── test_pipeline.py        # Pipeline testi
├── test_analyzer.py        # Analyzer testi
└── src/
    ├── config.py           # Yapılandırma ve eşikler
    ├── preprocessor.py     # Görüntü ön işleme
    ├── head_segmenter.py   # Kafa segmentasyonu (rembg/GrabCut/Otsu)
    ├── measurements.py     # SI, CVAI, geometrik ölçümler
    ├── classifier.py       # Şiddet sınıflandırması
    ├── face_analyzer.py    # MediaPipe yüz analizi
    ├── visualizer.py       # Görsel anotasyon
    ├── reporter.py         # PDF/HTML/metin raporu
    └── analyzer.py         # Ana pipeline koordinatörü
```

---

## 🧪 Algoritma

### 1. Segmentasyon (Kademeli)
1. **rembg** (U-Net AI) — en güvenilir
2. **GrabCut** (OpenCV) — orta güven
3. **Otsu Eşikleme** — hafif güven  
4. **Uyarlamalı Eşikleme** — son çare

### 2. Ölçüm
- `cv2.minAreaRect()` → Döndürülmüş minimum dikdörtgen → W, L
- `cv2.fitEllipse()` → Elips eksenleri (doğrulama)
- **CVAI**: Kafa AP eksenine 45°'lik diyagonaller

### 3. Sınıflandırma
- Kural tabanlı (Argenta skalası + Ott et al. CVAI eşikleri)
- SI ve CVAI bütünleşik değerlendirme
- Güven skoru: dairesellik × konvekslık × elips uyumu

---

## 📚 Referanslar

1. Argenta LC et al. *J Craniofac Surg*, 2004
2. Ott R et al. *Neuropediatrics*, 2007  
3. Loveday BP, de Chalain TB. *J Craniofac Surg*, 2001
4. WHO Baş Çevresi Standartları, 2006

---

## 🚀 GitHub'dan Hızlı Başlangıç

### 1. Depoyu Klonla

```bash
git clone https://github.com/ferhatcicek/dbts.git
cd dbts
```

### 2. Sanal Ortamı Kur (Windows)

```batch
setup_venv.bat
```

Bu script otomatik olarak:
- `venv/` klasörü oluşturur
- Tüm bağımlılıkları yükler
- `run.bat` kısayolunu oluşturur

### 3. Uygulamayı Başlat

```batch
run.bat
```

Tarayıcıda otomatik olarak `http://localhost:7860` adresi açılır.

---

## ⚠️ Yasal Uyarı

> Bu uygulama yalnızca **karar destek aracı** niteliğinde olup **tıbbi teşhis** amaçlı kullanılamaz. Elde edilen sonuçlar hiçbir koşulda uzman hekim muayene ve değerlendirmesinin yerini tutmaz. Sağlık sorunları için mutlaka yetkili sağlık kuruluşuna başvurun.

---

## 📁 Proje İçindeki GitHub İle İlgilendirilmeyecek Dosyalar

`.gitignore` sayesinde aşağıdakiler depoya eklenmez:

| Klasör / Dosya | Açıklama |
|---|---|
| `venv/` | Sanal ortam (klonlayan kişi kendisi kuruúr) |
| `__pycache__/` | Python derleme önbelleği |
| `*.onnx` | rembg AI model dosyaları (otomatik indirilir) |
| `.env` | Ortam değişkenleri |
| `*.log` | Log dosyaları |
| `run.bat` | `setup_venv.bat` tarafından oluşturulur |
