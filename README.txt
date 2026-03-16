# AI-Powered Medical Imaging Analyzer

Bu proje, beyin MR (MRI) görüntüleri üzerinden tümör tespiti "segmentasyon" yapan ve sonuçları analiz eden yapay zeka destekli bir medikal görüntüleme arayüzüdür. Derin öğrenme (U-Net) ve yerel büyük dil modelleri (LLM - Ollama) kullanılarak geliştirilmiştir.

## Özellikler

* Hassas Segmentasyon: PyTorch ile sıfırdan geliştirilen U-Net mimarisi sayesinde beyin MR görüntülerindeki tümörlü bölgelerin piksel bazında tespiti.
* Dinamik Görüntü İşleme: Orijinal MRI üzerine tespit edilen tümör maskesinin renkli olarak bindirilmesi.
* Otomatik Metrik Hesaplama: Tespit edilen tümörün yaklaşık piksel alanı ve merkez koordinatlarının otomatik hesaplanması.
* Yapay Zeka Radyolog Asistanı: Ollama (Llama 3) entegrasyonu sayesinde, elde edilen konum ve alan verileri kullanılarak otomatik ve profesyonel bir klinik ön rapor oluşturulması.
* Kullanıcı Dostu Arayüz: Streamlit kullanılarak geliştirilmiş, kolay kullanımlı web arayüzü.

## Kullanılan Teknolojiler

* Derin Öğrenme: PyTorch, Torchvision
* Model Mimarisi: U-Net Custom Implementation
* Bilgisayarlı Görüntü: OpenCV, Albumentations, Pillow
* Arayüz: Streamlit
* LLM Entegrasyonu: Ollama (Llama 3)
* Veri Manipülasyonu: NumPy, Pandas

## Proje Yapısı

├── app.py                 # Streamlit arayüz uygulaması ve ana çalışma dosyası
├── dataset.py             # PyTorch Dataset sınıfı ve veri artırımı (Albumentations) işlemleri
├── ollama_client.py       # Llama 3 ile iletişim kuran ve klinik rapor üreten modül
├── train.py               # U-Net modelinin eğitim betiği (DiceBCELoss, Adam, LR Scheduler)
├── unet.py                # U-Net modelinin PyTorch ile sıfırdan tasarımı
├── utils.py               # Dice katsayısı, alan hesaplama ve görüntü bindirme fonksiyonları
├── requirements.txt       # Proje bağımlılıkları
└── models/
    └── unet_best.pth      # Eğitilmiş en iyi model ağırlıkları. Kullanıcı tarafından eklenmelidir

## Kurulum ve Çalıştırma

### 1. Repoyu Klonlayın

git clone https://github.com/kullaniciadiniz/Medical-Imaging-Analyzer.git
cd Medical-Imaging-Analyzer


### 2. Gerekli Kütüphaneleri Yükleyin
Sanal bir ortam virtual environment oluşturmanız tavsiye edilir:

pip install -r requirements.txt


### 3. Ollama Kurulumu 
AI Radyolog asistanının çalışabilmesi için sisteminizde [Ollama](https://ollama.com/)'nın kurulu olması gerekmektedir.

# Ollama'yı başlatın ve Llama 3 modelini indirin
ollama run llama3


### 4. Modeli Eğitme (Opsiyonel)
Kendi veri setinizle modeli sıfırdan eğitmek isterseniz:

python train.py --data /veri_seti_yolu --epochs 50 --batch 16 --lr 0.0001


### 5. Uygulamayı Başlatma
Hazır model ağırlıklarınızı `models/` klasörüne ekledikten sonra arayüzü başlatabilirsiniz:

streamlit run app.py


## Uyarı
Bu yazılım yalnızca eğitim ve araştırma amaçlı geliştirilmiştir. Ürettiği segmentasyon sonuçları veya klinik raporlar kesinlikle profesyonel tıbbi tavsiye yerine geçmez. Tüm teşhisler uzman bir radyolog tarafından doğrulanmalıdır.

## Geliştirici
Umut Barbaros BABAHAN, Yapay Zeka Mühendisliği, Ostim Teknik Üniversitesi.