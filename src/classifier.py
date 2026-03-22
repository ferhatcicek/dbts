"""
classifier.py - Brakisefali şiddeti sınıflandırıcısı

Hesaplanan ölçümlerden hareketle şiddet derecesi belirler.
Kural tabanlı sistem (kanıta dayalı klinik eşikler):

  Birincil: Sefalik İndeks (SI)
  İkincil : Kranyal Kasa Asimetri İndeksi (CVAI)
  Üçüncül : Simetri skoru ve dairesellik

Argenta Skalası (2004) ve güncel literatür baz alınmıştır.
"""

import numpy as np
import logging
from typing import Optional

from src.config import (
    CEPHALIC_INDEX_THRESHOLDS,
    CVAI_THRESHOLDS,
    SEVERITY_LABELS_TR,
    SEVERITY_RECOMMENDATIONS_TR,
    SEVERITY_COLORS_RGB,
)

logger = logging.getLogger(__name__)


# ============================================================
# Ana sınıflandırma fonksiyonu
# ============================================================

def classify(measurements: dict) -> dict:
    """
    Ölçüm sonuçlarından klinik sınıflandırma üretir.

    Parametreler:
        measurements : compute_all_measurements() çıktısı

    Döner:
        dict:
          - 'ci_severity'    : SI'ya göre şiddet kodu
          - 'cvai_severity'  : CVAI'ya göre şiddet kodu
          - 'overall'        : Bütünleşik şiddet kodu
          - 'label_tr'       : Türkçe etiket
          - 'recommendation' : Türkçe öneri metni
          - 'color_rgb'      : Renk (R, G, B)
          - 'confidence'     : Güven skoru (0-1)
          - 'scores'         : Alt alan skorları dict
    """
    ci   = measurements.get("cephalic_index", 0)
    cvai = measurements.get("cvai", 0)
    circ = measurements.get("circularity", 0)
    sym  = measurements.get("symmetry_score", 0)

    if ci == 0:
        return _unknown_result()

    ci_sev   = _classify_ci(ci)
    cvai_sev = _classify_cvai(cvai)
    overall  = _combine_severities(ci_sev, cvai_sev, sym)

    confidence = _calc_confidence(measurements)

    return {
        "ci_severity":   ci_sev,
        "cvai_severity": cvai_sev,
        "overall":       overall,
        "label_tr":      SEVERITY_LABELS_TR.get(overall, overall),
        "recommendation": SEVERITY_RECOMMENDATIONS_TR.get(overall, ""),
        "color_rgb":     SEVERITY_COLORS_RGB.get(overall, (128, 128, 128)),
        "confidence":    round(confidence, 2),
        "scores": {
            "cephalic_index":   ci,
            "cvai":             cvai,
            "circularity":      round(circ, 3),
            "symmetry":         round(sym, 3),
        },
    }


# ============================================================
# Şiddet sınıflandırma yardımcıları
# ============================================================

def _classify_ci(ci: float) -> str:
    """Sefalik İndeks değerine göre şiddet kodu döner."""
    for severity, (low, high) in CEPHALIC_INDEX_THRESHOLDS.items():
        if low <= ci < high:
            return severity
    return "severe"


def _classify_cvai(cvai: float) -> str:
    """CVAI değerine göre asimetri şiddet kodu döner."""
    for severity, (low, high) in CVAI_THRESHOLDS.items():
        if low <= cvai < high:
            return severity
    return "severe"


def _combine_severities(ci_sev: str, cvai_sev: str, symmetry: float) -> str:
    """
    SI ve CVAI şiddetlerini birleştirerek genel şiddet kodu üretir.

    Mantık:
      - Birincil kriter: SI şiddeti
      - CVAI ≥ "moderate" → en az bir derece ağırlaştır
      - Simetri < 0.70 → ek 1 basamak ağırlaştır
    """
    order = ["dolikosefali", "normal", "mild", "moderate", "severe"]

    ci_idx   = order.index(ci_sev) if ci_sev in order else 2
    cvai_idx = order.index(cvai_sev) if cvai_sev in order else 1

    # CVAI'nin etkisi: orta veya ağır ise bir derece ekle
    combined_idx = ci_idx
    if cvai_idx >= order.index("moderate"):
        combined_idx = min(combined_idx + 1, len(order) - 1)

    # Simetri çok düşükse bir daha arttır
    if symmetry < 0.65 and combined_idx < len(order) - 1:
        combined_idx += 1

    return order[combined_idx]


def _calc_confidence(measurements: dict) -> float:
    """
    Ölçümün güvenilirliğini tahmin eder (0-1 arası).

    Faktörler:
      - Dairesellik (yüksek → daha güvenilir)
      - Konvekslık
      - Elips uyumu (ellipse_ci ≈ cephalic_index)
      - Simetri tutarlılığı
    """
    scores = []

    # Dairesellik: 0.5+ iyi
    circ = measurements.get("circularity", 0)
    scores.append(min(circ / 0.75, 1.0))

    # Konvekslık: 0.85+ iyi
    conv = measurements.get("convexity", 0)
    scores.append(min(conv / 0.90, 1.0))

    # Elips CI ile minAreaRect CI arasındaki fark
    ci_main = measurements.get("cephalic_index", 0)
    ci_ell  = measurements.get("ellipse_ci", 0)
    if ci_main > 0:
        diff_ratio = 1 - abs(ci_main - ci_ell) / max(ci_main, 1)
        scores.append(max(0.0, min(diff_ratio, 1.0)))

    # Simetri skoru zaten 0-1
    sym = measurements.get("symmetry_score", 0)
    scores.append(sym)

    return float(np.mean(scores)) if scores else 0.5


def _unknown_result() -> dict:
    """Ölçüm yapılamadığında dönen boş sonuç."""
    return {
        "ci_severity":   "unknown",
        "cvai_severity": "unknown",
        "overall":       "unknown",
        "label_tr":      "Ölçüm Yapılamadı",
        "recommendation": (
            "Kafa bölgesi yeterince tespit edilemedi. Lütfen daha net, "
            "iyi aydınlatılmış ve doğru açıdan çekilmiş bir fotoğraf yükleyin."
        ),
        "color_rgb":     (128, 128, 128),
        "confidence":    0.0,
        "scores": {},
    }


# ============================================================
# Şiddet dereceleri için yardımcı bilgiler
# ============================================================

def get_ci_interpretation(ci: float) -> str:
    """SI degeri icin kisa yorum doner (ASCII-safe)."""
    if ci == 0:
        return "Hesaplanamadi"
    if ci < 75:
        return f"SI={ci:.1f}% - Dolikosefali (dar/uzun kafa)"
    if ci < 85:
        return f"SI={ci:.1f}% - Normal aralik"
    if ci < 90:
        return f"SI={ci:.1f}% - Hafif Brakisefali"
    if ci < 95:
        return f"SI={ci:.1f}% - Orta Siddetli Brakisefali"
    return f"SI={ci:.1f}% - Agir Brakisefali"


def get_cvai_interpretation(cvai: float) -> str:
    """CVAI degeri icin kisa yorum doner (ASCII-safe)."""
    if cvai == 0:
        return "Hesaplanamadi"
    if cvai < 3.5:
        return f"KKAI={cvai:.1f}% - Simetrik"
    if cvai < 6.25:
        return f"KKAI={cvai:.1f}% - Hafif Asimetri"
    if cvai < 8.75:
        return f"KKAI={cvai:.1f}% - Orta Asimetri (Plagiosefali riski)"
    return f"KKAI={cvai:.1f}% - Agir Asimetri (Plagiosefali)"
