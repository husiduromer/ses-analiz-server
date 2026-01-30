from flask import Flask, request, jsonify
import os
import librosa
import numpy as np
import soundfile as sf # Linux için gerekli

app = Flask(__name__)

UPLOAD_FOLDER = 'gelen_sesler'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/analiz', methods=['POST'])
def analiz_et():
    if 'ses' not in request.files:
        return jsonify({'sonuc_baslik': 'Hata', 'grafik': []}), 400
    
    dosya = request.files['ses']
    cihaz = request.form.get('tur', 'Genel')
    
    dosya_yolu = os.path.join(UPLOAD_FOLDER, dosya.filename)
    dosya.save(dosya_yolu)
    
    try:
        # 1. SESİ YÜKLE
        y, sr = librosa.load(dosya_yolu)
        
        # 2. DETAYLI ÖLÇÜMLER (Mühendislik Verileri)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y)) # Metalik Sürtünme / Cızırtı
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        ortalama_frekans = np.mean(cent)                     # Sesin Tonu (Kalın/Tiz)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        darbe_gücü = np.mean(onset_env)                      # Vuruş / Tıkırtı Şiddeti
        rms = np.mean(librosa.feature.rms(y=y))              # Ses Seviyesi (Volume)

        print(f"📊 {cihaz} -> ZCR: {zcr:.3f} | Freq: {ortalama_frekans:.0f} | Darbe: {darbe_gücü:.3f}")

        # 3. TEŞHİS MOTORU (Expert System Logic)
        baslik = "✅ DURUM NORMAL"
        detay = f"{cihaz} değerleri stabil görünüyor.\nHerhangi bir anormallik tespit edilmedi."
        renk = "YESIL"

        # --- A) SESSİZLİK KONTROLÜ ---
        if rms < 0.01:
            return jsonify({
                'sonuc_baslik': "SESSİZ / BEKLEMEDE",
                'sonuc_detay': "Ortam sesi çok düşük.\nCihaz çalışmıyor veya uzakta.",
                'renk_kodu': "GRI",
                'grafik': _grafik_yap(y)
            })

        # --- B) CİHAZ BAZLI ARIZA SENARYOLARI ---
        
        # 🧊 1. BUZDOLABI SENARYOLARI
        if cihaz == "Buzdolabı":
            # Senaryo: Tıkırtı (Fan veya Röle)
            if darbe_gücü > 1.4:
                baslik = "⚠️ MEKANİK TIKIRTI"
                renk = "KIRMIZI"
                detay = (
                    "Olası Arıza Sebepleri:\n"
                    "🔴 %60 - Fan Pervanesi Buza Çarpıyor\n"
                    "🟠 %30 - Termik/Röle Arızası\n"
                    "🟡 %10 - Motor Takozları Gevşemiş"
                )
            # Senaryo: Yüksek Uğultu (Motor Zorlanması)
            elif ortalama_frekans < 1000 and zcr > 0.06:
                baslik = "⚠️ MOTOR/KOMPRESÖR"
                renk = "TURUNCU"
                detay = (
                    "Olası Arıza Sebepleri:\n"
                    "🔴 %70 - Kompresör Aşırı Isınıyor\n"
                    "🟠 %20 - Kondenser Kirliliği (Hava Alamıyor)\n"
                    "🟡 %10 - Gaz Dolaşım Sorunu"
                )
            # Senaryo: Gaz Sesi (Tıslama)
            elif ortalama_frekans > 2500 and zcr > 0.1:
                baslik = "⚠️ GAZ SİSTEMİ"
                renk = "KIRMIZI"
                detay = (
                    "Olası Arıza Sebepleri:\n"
                    "🔴 %80 - Soğutucu Gaz Kaçağı\n"
                    "🟠 %20 - Genleşme Valfi Tıkanıklığı"
                )

        # 🧺 2. ÇAMAŞIR MAKİNESİ SENARYOLARI
        elif cihaz == "Çamaşır Mak.":
            # Senaryo: Güm Güm Vurma (Sıkma Sırasında)
            if darbe_gücü > 2.0 and ortalama_frekans < 800:
                baslik = "⚠️ KAZAN DENGESİZLİĞİ"
                renk = "KIRMIZI"
                detay = (
                    "Olası Arıza Sebepleri:\n"
                    "🔴 %50 - Yük Dengesiz (Yorgan vb.)\n"
                    "🟠 %30 - Amortisörler Patlak\n"
                    "🟡 %20 - Kazan Rulmanları Dağılmış"
                )
            # Senaryo: Islık Sesi / Kayış
            elif ortalama_frekans > 3000:
                baslik = "⚠️ KAYIŞ/POMPA SORUNU"
                renk = "TURUNCU"
                detay = (
                    "Olası Arıza Sebepleri:\n"
                    "🔴 %70 - Kayış Kaçırıyor (Eskimis)\n"
                    "🟠 %30 - Tahliye Pompasına Cisim Kaçmış"
                )

        # 🚗 3. ARABA SENARYOLARI
        elif cihaz == "Araba":
            # Senaryo: Metalik Şıkırtı (Motor bloğundan)
            if darbe_gücü > 1.5 and zcr > 0.15:
                baslik = "⚠️ MOTOR SİBOP SESİ"
                renk = "KIRMIZI"
                detay = (
                    "Olası Arıza Sebepleri:\n"
                    "🔴 %60 - Sibop/İtici (Lifter) Arızası\n"
                    "🟠 %30 - Yağ Seviyesi Kritik Düşük\n"
                    "🟡 %10 - Enjektör Problemi"
                )
            # Senaryo: Kayış Ötmesi
            elif ortalama_frekans > 4000:
                baslik = "⚠️ V-KAYIŞI SESİ"
                renk = "TURUNCU"
                detay = (
                    "Olası Arıza Sebepleri:\n"
                    "🔴 %90 - Alternatör Kayışı Gevşek/Eskimiş\n"
                    "🟠 %10 - Bilya Dağılması"
                )

        # 🏍️ 4. MOTOSİKLET SENARYOLARI
        elif cihaz == "Motosiklet":
             if zcr > 0.4:
                baslik = "⚠️ EGZOZ/BLOK SESİ"
                renk = "KIRMIZI"
                detay = (
                    "Olası Arıza Sebepleri:\n"
                    "🔴 %50 - Egzoz Patlak/Conta Yanık\n"
                    "🟠 %40 - Eksantrik Zinciri Gevşek\n"
                    "🟡 %10 - Sübap Ayarı Bozuk"
                )

        # 📺 5. GENEL ELEKTRONİK (TV vb.)
        else: 
            if zcr > 0.2:
                baslik = "⚠️ ELEKTRONİK GÜRÜLTÜ"
                renk = "TURUNCU"
                detay = "Cihazda bobin vızıltısı (Coil Whine) veya kondansatör sorunu olabilir (%80 İhtimal)."

        # GRAFİK VERİSİ HAZIRLA
        return jsonify({
            'sonuc_baslik': baslik,
            'sonuc_detay': detay,
            'renk_kodu': renk,
            'grafik': _grafik_yap(y)
        })
        
    except Exception as e:
        print(f"HATA: {e}")
        return jsonify({'sonuc_baslik': "Hata", 'sonuc_detay': str(e), 'renk_kodu': "GRI", 'grafik': []})

# Grafik verisini hazırlayan yardımcı fonksiyon
def _grafik_yap(y):
    adim = len(y) // 50
    if adim < 1: adim = 1
    return np.abs(y[::adim]).tolist()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
