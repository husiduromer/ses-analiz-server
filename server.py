from flask import Flask, request, jsonify
import os
import librosa
import numpy as np

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
        
        # 2. ÖLÇÜMLER
        zcr = np.mean(librosa.feature.zero_crossing_rate(y)) # Cızırtı
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        ortalama_frekans = np.mean(cent)                     # Sesin tonu (Kalın/Tiz)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        darbe_gücü = np.mean(onset_env)                      # Vuruş/Tıkırtı
        rms = np.mean(librosa.feature.rms(y=y))              # Ses Şiddeti

        print(f"📊 {cihaz} -> ZCR: {zcr:.3f} | Frekans: {ortalama_frekans:.0f} | Darbe: {darbe_gücü:.3f} | Ses Gücü: {rms:.4f}")

        # 3. KARAR MEKANİZMASI
        # Varsayılan: Her şey yolunda
        baslik = "✅ DURUM NORMAL"
        detay = f"{cihaz} stabil çalışıyor."
        renk = "YESIL"

        # Önce SESSİZLİK kontrolü (Boşa panik yapmasın)
        if rms < 0.01:
            baslik = "SESSİZ / BEKLEMEDE"
            detay = "Cihaz çalışmıyor veya ortam çok sessiz."
            renk = "GRI"

        # --- BUZDOLABI İÇİN ÖZEL AYAR (GÜNCELLENDİ) ---
        elif cihaz == "Buzdolabı":
            # 1. Tıkırtı (Fan çarpması vb.) - Eşik 1.2 -> 1.5'e çıktı (Daha zor tetiklenir)
            if darbe_gücü > 1.5: 
                baslik = "⚠️ MEKANİK ARIZA"
                detay = "Pervane çarpması veya röle tıkırtısı olabilir."
                renk = "KIRMIZI"
            
            # 2. Uğultu (Motor Zorlanması) - ZCR 0.03 -> 0.08'e çıktı (Artık sadece ses kalın diye hata vermez, cızırtı da lazım)
            elif ortalama_frekans < 1500 and zcr > 0.08:
                baslik = "⚠️ MOTOR ZORLANIYOR"
                detay = "Kompresör sarsıntılı çalışıyor olabilir."
                renk = "TURUNCU"
                
            # 3. Gaz Sesi (Tıslama) - ZCR 0.05 -> 0.12'ye çıktı (Çok belirgin tıslama lazım)
            elif ortalama_frekans > 3000 and zcr > 0.12:
                baslik = "⚠️ GAZ SİSTEMİ"
                detay = "Gaz akışında tıkanıklık veya kaçak sesi."
                renk = "KIRMIZI"

        # --- DİĞER CİHAZLAR ---
        elif cihaz == "Motosiklet":
            if zcr > 0.35: # Motor zaten gürültülüdür, eşiği çok yüksek tuttuk
                baslik = "⚠️ EGZOZ/MOTOR SORUNU"
                detay = "Ses normalden çok daha metalik/patlak."
                renk = "KIRMIZI"
        
        else: # Genel Mod
            if zcr > 0.15:
                baslik = "⚠️ YÜKSEK GÜRÜLTÜ"
                detay = "Normalden fazla sürtünme sesi var."
                renk = "TURUNCU"

        # Grafik verisi
        adim = len(y) // 50
        if adim < 1: adim = 1
        grafik_verisi = np.abs(y[::adim]).tolist() 

        return jsonify({
            'sonuc_baslik': baslik,
            'sonuc_detay': detay,
            'renk_kodu': renk,
            'grafik': grafik_verisi
        })
        
    except Exception as e:
        print(f"HATA: {e}")
        return jsonify({'sonuc_baslik': "Hata", 'sonuc_detay': str(e), 'renk_kodu': "GRI", 'grafik': []})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)