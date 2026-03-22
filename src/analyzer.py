"""
analyzer.py - Ana analiz pipeline koordinatörü

Tüm modülleri bir araya getirir:
  1. Ön işleme
  2. Segmentasyon
  3. Ölçüm
  4. Sınıflandırma
  5. Görselleştirme
  6. Raporlama

Desteklenen görünüm türleri:
  - 'top'   : Üstten (tepe) görünüm — birincil analiz (Brakisefali)
  - 'front' : Önden görünüm — yüz/kafa oranları
  - 'side'  : Yandan görünüm — profil (sınırlı destek)
"""

import numpy as np
import logging
from typing import Optional

from src.preprocessor    import preprocess, check_image_quality, load_image_from_numpy
from src.head_segmenter  import segment_head
from src.measurements    import compute_all_measurements
from src.classifier      import classify
from src.visualizer      import create_annotated_image, create_gauge_image
from src.reporter        import generate_pdf_report, generate_text_report
from src.face_analyzer   import FaceAnalyzer

logger = logging.getLogger(__name__)

# Paylaşılan FaceAnalyzer örneği (MediaPipe yükleme maliyetini amortize eder)
_face_analyzer: Optional[FaceAnalyzer] = None


def get_face_analyzer() -> FaceAnalyzer:
    global _face_analyzer
    if _face_analyzer is None:
        _face_analyzer = FaceAnalyzer()
    return _face_analyzer


# ============================================================
# Ana analiz fonksiyonu
# ============================================================

def analyze(
    img_rgb: np.ndarray,
    view: str = "top",
    patient_info: Optional[dict] = None,
) -> dict:
    """
    Tam analiz pipeline'ını çalıştırır.

    Parametreler:
        img_rgb      : Ön işlenmemiş RGB görüntü
        view         : 'top' | 'front' | 'side'
        patient_info : İsteğe bağlı hasta meta verisi

    Döner:
        dict:
          - 'success'        : bool
          - 'error'          : hata mesajı (varsa)
          - 'quality'        : görüntü kalite kontrolü
          - 'measurements'   : geometrik ölçümler
          - 'classification' : şiddet sınıflandırması
          - 'annotated_img'  : RGB annotasyonlu görüntü
          - 'gauge_img'      : RGB gösterge görüntüsü
          - 'text_report'    : metin raporu
          - 'pdf_path'       : PDF dosya yolu (varsa)
          - 'seg_result'     : segmentasyon meta verisi
    """
    result = {
        "success":        False,
        "error":          None,
        "quality":        {},
        "measurements":   {},
        "classification": {},
        "annotated_img":  None,
        "gauge_img":      None,
        "text_report":    "",
        "pdf_path":       None,
        "seg_result":     {},
    }

    try:
        # 1. Görüntü kalite kontrolü
        quality = check_image_quality(img_rgb)
        result["quality"] = quality

        # 2. Ön işleme
        pre = preprocess(img_rgb)
        img_working = pre["resized"]    # (h, w, 3) RGB

        # 3. Görünüm türüne göre analiz dallanması
        if view == "top":
            result = _analyze_top_view(img_working, result, patient_info)
        elif view == "front":
            result = _analyze_front_view(img_working, result, patient_info)
        elif view == "side":
            result = _analyze_side_view(img_working, result, patient_info)
        else:
            raise ValueError(f"Bilinmeyen görünüm türü: {view}")

    except Exception as e:
        logger.exception(f"Analiz hatası: {e}")
        result["success"] = False
        result["error"] = str(e)

    return result


# ============================================================
# Üstten görünüm analizi (birincil yöntem)
# ============================================================

def _analyze_top_view(img_rgb: np.ndarray, result: dict, patient_info) -> dict:
    """Kafanın üstten görüntüsünden CI ve CVAI hesaplar."""

    # Kafa segmentasyonu
    seg = segment_head(img_rgb)
    result["seg_result"] = {
        "mask":       seg.get("mask"),
        "method":     seg.get("method"),
        "confidence": seg.get("confidence"),
        "success":    seg.get("success"),
    }

    if not seg["success"]:
        result["error"] = (
            "Kafa bölgesi tespit edilemedi. Lütfen daha net ve iyi "
            "aydınlatılmış bir üstten çekim fotoğrafı yükleyin."
        )
        # Yine de boş ölçüm ve sınıflandırma oluştur
        result["measurements"]   = {}
        result["classification"] = {}
        annotated = _make_error_overlay(img_rgb, result["error"])
        result["annotated_img"]  = annotated
        return result

    # Ölçümler
    measurements = compute_all_measurements(seg["contour"], seg["mask"])
    result["measurements"] = measurements

    # Sınıflandırma
    classification = classify(measurements)
    result["classification"] = classification

    # Görselleştirme
    annotated = create_annotated_image(img_rgb, seg, measurements, classification)
    result["annotated_img"] = annotated

    gauge = create_gauge_image(classification)
    result["gauge_img"] = gauge

    # Raporlar
    result["text_report"] = generate_text_report(measurements, classification, patient_info)
    result["pdf_path"]    = generate_pdf_report(
        img_rgb, annotated, measurements, classification, patient_info
    )
    result["success"] = True
    return result


# ============================================================
# Önden görünüm analizi
# ============================================================

def _analyze_front_view(img_rgb: np.ndarray, result: dict, patient_info) -> dict:
    """MediaPipe ile önden yüz/kafa ölçümleri."""

    fa = get_face_analyzer()
    face_data = fa.analyze(img_rgb, view="front")

    if not face_data.get("success"):
        result["error"] = (
            f"Yüz tespiti başarısız: {face_data.get('error', 'bilinmeyen hata')}. "
            "Lütfen net bir önden fotoğraf yükleyin."
        )
        result["annotated_img"] = _make_error_overlay(img_rgb, result["error"])
        return result

    # Frontal ölçümleri measurements dict'e köprüle
    # Not: CI hesabı üstten görüntü gerektirir; frontal sadece yardımcı ölçüm
    measurements = {
        "cephalic_index": 0,
        "cvai":           0,
        "head_width_px":  face_data.get("head_width_px", 0),
        "face_height_px": face_data.get("face_height_px", 0),
        "face_index":     face_data.get("face_index", 0),
        "asymmetry_pct":  face_data.get("asymmetry_pct", 0),
        "symmetry_score": max(0, 1 - face_data.get("asymmetry_pct", 0) / 100),
        "circularity":    0,
        "convexity":      0,
        "view":           "front",
    }
    result["measurements"] = measurements

    # Frontal görünümde CI hesaplanamaz; sadece asimetri skoru
    classification = {
        "overall":       "unknown",
        "label_tr":      "Üstten Görünüm Gerekli (CI hesaplanamadı)",
        "ci_severity":   "unknown",
        "cvai_severity": "unknown",
        "recommendation": (
            "Sefalik İndeks (SI) hesabı için kafanın üstten çekilmiş "
            "fotoğrafı gerekmektedir. Mevcut analiz yalnızca yüz "
            "asimetri değerlendirmesi içermektedir. "
            f"Yüz asimetrisi: {face_data.get('asymmetry_pct', 0):.1f}%"
        ),
        "confidence": 0.6,
        "scores": {},
    }
    result["classification"] = classification

    # Görsel: üzerine landmark çiz
    annotated = fa.draw_landmarks(img_rgb, face_data)
    result["annotated_img"] = annotated

    result["text_report"] = generate_text_report(measurements, classification, patient_info)
    result["success"] = True
    return result


# ============================================================
# Yandan görünüm analizi
# ============================================================

def _analyze_side_view(img_rgb: np.ndarray, result: dict, patient_info) -> dict:
    """Yandan görünüm: segmentasyon + profil bilgisi."""

    # Lateral görünümde de segmentasyon mantıklı sonuç verir
    seg = segment_head(img_rgb)
    result["seg_result"] = seg

    measurements = {}
    if seg["success"] and seg["contour"] is not None:
        import cv2
        import math
        cnt = seg["contour"]
        min_rect = cv2.minAreaRect(cnt)
        (cx, cy), (rw, rh), angle = min_rect
        # Lateral: AP uzunluğu uzun kenar, yükseklik kısa kenar
        head_length = max(rw, rh)
        head_height = min(rw, rh)
        measurements = {
            "head_length_px": round(float(head_length), 1),
            "head_height_px": round(float(head_height), 1),
            "profile_ratio":  round(float(head_height / head_length), 3) if head_length > 0 else 0,
            "cephalic_index": 0,
            "cvai":           0,
            "symmetry_score": 0,
            "circularity":    0,
            "view":           "side",
        }

    result["measurements"] = measurements

    classification = {
        "overall":       "unknown",
        "label_tr":      "Yandan Görünüm — CI İçin Üstten Fotoğraf Gerekli",
        "ci_severity":   "unknown",
        "cvai_severity": "unknown",
        "recommendation": (
            "Brakisefali tanısı için kafa üstten (tepe noktasından) "
            "fotoğraflanmalıdır. Bu yandan görünüm fotoğrafından yalnızca "
            "kafa profil oranı hesaplanabilmektedir."
        ),
        "confidence": 0.3,
        "scores": {},
    }
    result["classification"] = classification

    if seg["success"]:
        annotated = create_annotated_image(img_rgb, seg, measurements, classification)
    else:
        annotated = img_rgb.copy()
    result["annotated_img"] = annotated

    result["text_report"] = generate_text_report(measurements, classification, patient_info)
    result["success"] = True
    return result


# ============================================================
# Yardımcı fonksiyonlar
# ============================================================

def _make_error_overlay(img_rgb: np.ndarray, message: str) -> np.ndarray:
    """Hata mesajını görüntü üzerine yazar."""
    import cv2
    vis = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = vis.shape[:2]

    # Yarı saydam kırmızı bant
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 160), -1)
    cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)
    cv2.putText(vis, "HATA: Segmentasyon basarisiz", (10, 32),
               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Alt mesaj
    words = message.split()
    line, lines = [], []
    for word in words:
        if len(" ".join(line + [word])) < 80:
            line.append(word)
        else:
            lines.append(" ".join(line))
            line = [word]
    if line:
        lines.append(" ".join(line))

    for i, l in enumerate(lines[:4]):
        cv2.putText(vis, l, (10, 70 + i * 22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 255), 1, cv2.LINE_AA)

    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
