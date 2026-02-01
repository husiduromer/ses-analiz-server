from flask import Flask, request, jsonify
import os
import librosa
import numpy as np
import soundfile as sf

app = Flask(__name__)

UPLOAD_FOLDER = 'gelen_sesler'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def grafik_verisi_hazirla(y):
    try:
        adim = len(y) // 50
        if adim < 1: adim = 1
        return np.abs(y[::adim]).tolist()
    except:
        return []

@app.route('/analiz', methods=['POST'])
def analiz_et():
    if 'ses' not in request.files:
        return jsonify({'sonuc_baslik': 'Hata', 'grafik': []}), 400
    
    dosya = request.files['ses']
    cihaz = request.form.get('tur', 'Genel')
    
    dosya_yolu = os.path.join(UPLOAD_FOLDER, dosya.filename)
    dosya.save(dosya_yolu)
    
    try:
        # --- İŞTE SİHİRLİ DOKUNUŞ BURADA ---
        # duration=5 : Sadece ilk 5 saniyeyi yükle (RAM Tasarrufu)
        y, sr = librosa.load(dosya_yolu, duration=5)
        
        # 2. ÖLÇÜMLER
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        ortalama_frekans = np.mean(cent)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        darbe_gücü = np.mean(onset_env)
        rms = np.mean(librosa.feature.rms(y=y))

        print(f"📊 {cihaz} -> ZCR: {zcr:.3f} | Freq: {ortalama_frekans:.0f} | Darbe: {darbe_gücü:.3f}")

        # 3. SENARYOLAR
        baslik = "✅ DURUM NORMAL"
        detay = f"{cihaz} değerleri stabil."
        renk = "YESIL"

        # SESSİZLİK
        if rms < 0.01:
            baslik = "SESSİZ / BEKLEMEDE"
            detay = "Ses seviyesi analiz için çok düşük."
            renk = "GRI"
        
        # BUZDOLABI
        elif cihaz == "Buzdolabı":
            if darbe_gücü > 1.4:
                baslik = "⚠️ MEKANİK TIKIRTI"
                renk = "KIRMIZI"
                detay = "Fan pervanesi çarpıyor (%60) veya röle arızası (%30)."
            elif ortalama_frekans < 1000 and zcr > 0.06:
                baslik = "⚠️ MOTOR ZORLANIYOR"
                renk = "TURUNCU"
                detay = "Kompresör aşırı ısınıyor veya takozlar eskimiş."
            elif ortalama_frekans > 2500 and zcr > 0.1:
                baslik = "⚠️ GAZ SİSTEMİ"
                renk = "KIRMIZI"
                detay = "Soğutucu gaz akışında tıkanıklık veya kaçak."

        # ÇAMAŞIR MAKİNESİ
        elif cihaz == "Çamaşır Mak.":
            if darbe_gücü > 2.0 and ortalama_frekans < 800:
                baslik = "⚠️ KAZAN DENGESİZLİĞİ"
                renk = "KIRMIZI"
                detay = "Yük dengesiz (%50) veya amortisörler patlak (%30)."
            elif ortalama_frekans > 3000:
                baslik = "⚠️ KAYIŞ/POMPA"
                renk = "TURUNCU"
                detay = "Kayış kaçırıyor veya pompaya cisim kaçmış."

        # ARABA
        elif cihaz == "Araba":
            if darbe_gücü > 1.5 and zcr > 0.15:
                baslik = "⚠️ MOTOR SİBOP SESİ"
                renk = "KIRMIZI"
                detay = "Sibop iticileri arızalı (%60) veya yağ seviyesi düşük."
            elif ortalama_frekans > 4000:
                baslik = "⚠️ V-KAYIŞI SESİ"
                renk = "TURUNCU"
                detay = "Alternatör kayışı gevşek veya bilya dağılmış."

        # GENEL
        else:
            if zcr > 0.2:
                baslik = "⚠️ GENEL GÜRÜLTÜ"
                renk = "TURUNCU"
                detay = "Cihazda normalden fazla sürtünme sesi var."

        return jsonify({
            'sonuc_baslik': baslik,
            'sonuc_detay': detay,
            'renk_kodu': renk,
            'grafik': grafik_verisi_hazirla(y)
        })
        
    except Exception as e:
        print(f"HATA: {e}")
        return jsonify({
            'sonuc_baslik': "Sunucu Hatası", 
            'sonuc_detay': "Sunucu yoğun, lütfen daha kısa kayıt yapın.", 
            'renk_kodu': "GRI", 
            'grafik': []
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
