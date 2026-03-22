"""
Deformasyonel Brakisefali Tespit Sistemi
Ana yapılandırma ve sabitler modülü
"""

# ============================================================
# TANISAL EŞİK DEĞERLERİ (Klinik literatüre dayanmaktadır)
# ============================================================

# Sefalik İndeks (SI) = (Kafa Genişliği / Kafa Uzunluğu) × 100
# Referans: Argenta sınıflaması ve WHO standartları
CEPHALIC_INDEX_THRESHOLDS = {
    "dolikosefali":     (0,    75.0),   # Dar/uzun kafa
    "normal":           (75.0, 85.0),   # Normal aralık
    "mild":             (85.0, 90.0),   # Hafif brakisefali
    "moderate":         (90.0, 95.0),   # Orta brakisefali
    "severe":           (95.0, float("inf")),  # Ağır brakisefali
}

# Kranyal Kasa Asimetri İndeksi (KKAI) = |D1 - D2| / D2 × 100
# Referans: Ott ve ark. (2007), Loveday & de Chalain (2001)
CVAI_THRESHOLDS = {
    "normal":           (0,    3.5),    # Simetrik
    "mild":             (3.5,  6.25),   # Hafif asimetri
    "moderate":         (6.25, 8.75),   # Orta asimetri
    "severe":           (8.75, float("inf")),   # Ağır asimetri
}

# ============================================================
# RENKLENDİRME (BGR formatında - OpenCV)
# ============================================================

SEVERITY_COLORS_BGR = {
    "dolikosefali": (255, 165, 0),   # Turuncu
    "normal":       (0, 200, 0),     # Yeşil
    "mild":         (0, 200, 255),   # Sarı
    "moderate":     (0, 100, 255),   # Turuncu
    "severe":       (0, 0, 220),     # Kırmızı
}

SEVERITY_COLORS_RGB = {
    "dolikosefali": (255, 165, 0),
    "normal":       (0, 200, 0),
    "mild":         (255, 200, 0),
    "moderate":     (255, 100, 0),
    "severe":       (220, 0, 0),
}

# ============================================================
# TÜRKÇe ETİKETLER
# ============================================================

SEVERITY_LABELS_TR = {
    "dolikosefali": "Dolikosefali (Uzun Kafa)",
    "normal":       "Normal",
    "mild":         "Hafif Brakisefali",
    "moderate":     "Orta Şiddetli Brakisefali",
    "severe":       "Ağır Brakisefali",
}

SEVERITY_RECOMMENDATIONS_TR = {
    "dolikosefali": (
        "Sefalik indeks düşük bulunmuştur. Bu durum kafanın normalden "
        "uzun ve dar olduğuna işaret edebilir. Bir çocuk nöroloji veya "
        "kraniyofasiyal uzmanına danışmanız önerilir."
    ),
    "normal": (
        "Ölçüm değerleri normal aralıkta görünmektedir. Kafa şekli "
        "genel olarak simetrik ve orantılı. Düzenli pediatrik kontrolleri "
        "sürdürmeye devam edin."
    ),
    "mild": (
        "Hafif düzeyde brakisefali bulgusu tespit edilmiştir. Bebek uyurken "
        "baş pozisyonunu değiştirmek yararlı olabilir. Bir pediatrist veya "
        "fizyoterapist ile görüşmeniz tavsiye edilir."
    ),
    "moderate": (
        "Orta şiddetli brakisefali bulgusu tespit edilmiştir. Koruyucu "
        "başlık (kranyal ortez/bant) tedavisi değerlendirilebilir. "
        "Lütfen en kısa sürede bir kraniyofasiyal uzman veya "
        "çocuk nöroşirurjiyenine başvurun."
    ),
    "severe": (
        "Ağır düzeyde brakisefali bulgusu tespit edilmiştir. Acil tıbbi "
        "değerlendirme gereklidir. Lütfen derhal bir kraniyofasiyal uzman "
        "ya da çocuk nöroşirurjiyenine başvurun. Erken müdahale "
        "kritik önem taşımaktadır."
    ),
}

# ============================================================
# GÖRÜNTÜ İŞLEME AYARLARI
# ============================================================

# Hedef analiz boyutu (piksel)
TARGET_IMAGE_SIZE = 1024

# GrabCut yineleme sayısı (arttırca daha doğru ama yavaş)
GRABCUT_ITERATIONS = 5

# Minimum kafa alanı oranı (görüntü alanının yüzdesi)
MIN_HEAD_AREA_RATIO = 0.05
MAX_HEAD_AREA_RATIO = 0.98

# Minimum yuvarlıklık skoru (0-1 arası; 1=mükemmel çember)
MIN_CIRCULARITY = 0.35

# ============================================================
# GÖRÜNÜM TÜRÜ AYARLARI
# ============================================================

VIEW_TYPES = {
    "top":      "Üstten Görünüm (Tepe) - Brakisefali Analizi",
    "front":    "Önden Görünüm (Frontal) - Yüz Oranı Analizi",
    "side":     "Yandan Görünüm (Lateral) - Profil Analizi",
}

# ============================================================
# UYGULAMA AYARLARI
# ============================================================

APP_TITLE = "Deformasyonel Brakisefali Tespit Sistemi"
APP_VERSION = "1.0.0"
APP_AUTHOR = "AI Klinik Destek Aracı"

MEDICAL_DISCLAIMER = (
    "⚠️ YASAL UYARI: Bu uygulama yalnızca karar destek aracı niteliğinde olup "
    "tıbbi teşhis amaçlı kullanılamaz. Elde edilen sonuçlar hiçbir koşulda "
    "uzman hekim muayene ve değerlendirmesinin yerini tutmaz. Sağlık sorunları "
    "için mutlaka yetkili sağlık kuruluşuna başvurun."
)
