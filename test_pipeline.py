"""Test script - pipeline doğrulama"""
import sys
import numpy as np
import cv2

sys.path.insert(0, ".")

# Sentetik 'brakisefalik kafa' oluştur
# Elips: yatay yarı çap 180px (genişlik=W), dikey yarı çap 130px (uzunluk=L)
# CI = W/L * 100 = 180/130 * 100 = 138 -> Resizolanmış boyuta dikkat et
# minAreaRect ile: W=min(w,h), L=max(w,h) -> min=260, max=360 -> CI=260/360*100=72
# Demek istediğim: genişlik=360 (yatay çap), uzunluk=260 (dikey çap)
# CI = 260/360 * 100 = 72 -> normalin altı (dolikosefali)
# Brakisefali için: geniş (yatay) > uzun (dikey) -> yatay > dikey
# Kafa yandan basık = W > L = CI > 100 olmaz, max 100 mantıksal sınır...
# Kliniklde: yatay genişlik (W) / ön-arka uzunluk (L)
# Brakisefali = yatay > ön-arka, CI > 85
# Elips: yatay=180 (W), dikey=130 (L) -> CI = 180/130 * 100 = 138? Hayır, normalleştirilmiş

img = np.ones((500, 500, 3), dtype=np.uint8) * 200  # Açık gri arka plan
# Brakisefalik elips: geniş yatay (280px çap=W) kısa dikey (200px çap=L)
# CI = W/L = 280/200 * 100 = 140 mantıksız. 
# Gerçekte CI 75-110 arası değişir; piksel oranı doğrudan CI = W/L * 100
# Yatay çap 160, dikey çap 220 -> W=320, L=440 -> CI = 320/440*100 = 72 (Dolikosefali)
# Yatay çap 200, dikey çap 140 -> W=400 (geniş), L=280 (kısa AP) -> CI=400/280*100=142??
# O zaman minAreaRect: kısa kenar = min = dikey = 280, uzun kenar = max = yatay = 400
# CI = kısa/uzun * 100 = 280/400 * 100 = 70 -> YANLIŞ
# Formülü düzeltmeliyim: CI = GENIŞLIK(yatay) / UZUNLUK(AP/dikey) * 100
# minAreaRect bize sadece w ve h verir; hangisinin W hangisinin L olduğunu belirlemek şart
# Elips açısına bakarak: angle=0 ise dikey eksen AP
# Pratikte: CI = short/long * 100 her zaman < 100 YANLIŞ
# Klinik: W > L → brakisefali (CI > 80), bu durumda yatay > dikey
# Sentetik test: yatay çap=150 px (L=AP), dikey çap=180 px (W=temporal) -> CI=180/150*100=120?
# Kafanın üstten görünümünde:
#   AP uzunluk = ön-arka boyut = dikey eksen
#   Temporal genişlik = yatay eksen
#   Brakisefali = temporal > AP, yani yatay > dikey
# Elips: yatay yarıçap=190 (W=380), dikey yarıçap=120 (L=240) -> CI = W/L = 380/240*100 = 158? Hayır
# Piksel ölçümdeki oran: CI = W_piksel / L_piksel * 100 = 380/240 * 100 = 158 olmaz
# Çünkü brakisefali için W > L ama CI 85-95 arası
# Demek ki CI doğrudan piksel oranı değil: CI = W/L * 100 ve W < L her zaman
# YANLIŞ - brakisefali = W > L ise CI > 100, normal = W < L ise CI < 100
# AMA klinikte CI = W/max_diagonal olarak ölçülmez
# Doğru tanım: CI = (kafa maksimum genişliği) / (kafa maksimum uzunluğu) × 100
# Ve brakisefali için CI > 80... yani W > L ve CI > 80
# CI 100 olamaz; çünkü kafa küre değil
# Aslında bakıldığında: CI > 85 brakisefali; CI normalde 75-85
# Yani bir kafa 10cm geniş 12cm uzun ise CI = 10/12*100 = 83 (normal)
# Yatay basık kafa: 12cm geniş 11cm uzun -> CI = 12/11*100 = 109 - bu brakisefali nasıl?
# Tekrar: CI = BİÇİM BOZUKLUĞU, genişlik UZUNLUĞU geçiyor
# YEP: CI = W/L * 100, normal W < L, brakisefali W yaklaşık L veya W > L
# Normal kafada CI 75-85 demek: W = 75-85% of L, yani L > W
# Brakisefali CI=90 demek: W = 90% of L -> hala L > W, ama daha az fark
# CI=100: W = L (küresel kafa)
# CI > 100 teorik olarak mümkün değil çünkü kafalar küre içi

# Kısaca: minAreaRect -> kısa kenar = W, uzun kenar = L -> CI = W/L * 100
# Bu doğru! Normal: CI 75-85 -> w/l = 0.75-0.85 -> L > W
# Brakisefali: CI > 85 -> W/L > 0.85 -> kafa daha yuvarlak/basık

# Sentetik: W=320, L=400 -> CI = 320/400*100 = 80 (normal)
# Sentetik brakisefali: W=350, L=380 -> CI = 350/380*100 = 92 (moderate)
# Elips yatay yarıçap=175 (W=350), dikey yarıçap=190 (L=380)
# Ama bu durumda DIKEY daha büyük, yani uzun eksen dikey
# minAreaRect bu durumda: dikey kenar=380 (L), yatay kenar=350 (W) -> CI=350/380*100=92 ✓

cv2.ellipse(img, (250, 250), (175, 190), 0, 0, 360, (80, 60, 50), -1)

from src.preprocessor import preprocess, check_image_quality
from src.head_segmenter import segment_head
from src.measurements import compute_all_measurements
from src.classifier import classify

pre = preprocess(img)
working = pre["resized"]

print("Kalite kontrolu...")
q = check_image_quality(working)
print(f"  Uyarilar: {q['warnings']}")

print()
print("Segmentasyon...")
seg = segment_head(working)
print(f"  Yontem  : {seg['method']}")
print(f"  Basari  : {seg['success']}")
print(f"  Guven   : {seg['confidence']}")

if seg["success"]:
    print()
    print("Olcumler...")
    m = compute_all_measurements(seg["contour"], seg["mask"])
    ci   = m["cephalic_index"]
    cvai = m["cvai"]
    sym  = m["symmetry_score"]
    print(f"  Sefalik Indeks (SI): {ci:.2f}%")
    print(f"  CVAI               : {cvai:.2f}%")
    print(f"  Simetri            : {sym*100:.1f}%")
    print(f"  Dairesellik        : {m['circularity']:.3f}")
    print(f"  W (px)             : {m['width_px']:.0f}")
    print(f"  L (px)             : {m['length_px']:.0f}")

    print()
    print("Siniflandirma...")
    c = classify(m)
    print(f"  Sonuc : {c['label_tr']}")
    print(f"  Guven : {c['confidence']*100:.0f}%")
    print(f"  Oneri : {c['recommendation'][:100]}")
    print()
    print("=== PIPELINE TESTI BASARILI ===")
else:
    print("[UYARI] Segmentasyon basarisiz")
