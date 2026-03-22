"""
visualizer.py - Ölçüm sonuçlarının görüntü üzerine bindirimi

Anotasyonlar:
  - Kafa konturu (renk şiddet derecesine göre)
  - Minimum alanı döndürülmüş dikdörtgen (W, L ölçümleri)
  - Elips uydurması
  - Diyagonal çizgiler (D1, D2 - CVAI için)
  - Ölçüm değerleri metin etiketi
  - Şiddet göstergesi (renk kodlu)
  - Güven çubuğu
"""

import cv2
import numpy as np
import math
from typing import Optional, Tuple, List
import logging

from src.config import SEVERITY_COLORS_BGR, SEVERITY_LABELS_TR

logger = logging.getLogger(__name__)


# ============================================================
# Ana görselleştirme fonksiyonu
# ============================================================

def create_annotated_image(
    img_rgb: np.ndarray,
    seg_result: dict,
    measurements: dict,
    classification: dict,
) -> np.ndarray:
    """
    Segmentasyon, ölçüm ve sınıflandırma sonuçlarını görüntü üzerine bindirir.

    Döner: RGB annotasyonlu görüntü
    """
    # BGR çalışma kopyası
    vis = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    severity     = classification.get("overall", "unknown")
    color_bgr    = SEVERITY_COLORS_BGR.get(severity, (180, 180, 180))
    confidence   = classification.get("confidence", 0.0)

    # 1. Kafa maskesini yarı saydam bindirme
    mask = seg_result.get("mask")
    if mask is not None:
        _overlay_mask(vis, mask, color_bgr, alpha=0.18)

    contour = seg_result.get("contour")
    if contour is not None:
        # 2. Kafa konturu
        cv2.drawContours(vis, [contour], -1, color_bgr, 2)

        # 3. Minimum alanı çevreleyen döndürülmüş dikdörtgen
        min_rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(min_rect).astype(np.int32)
        cv2.drawContours(vis, [box], 0, (255, 255, 0), 1)

        # 4. Elips
        if measurements.get("ellipse") and len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(vis, ellipse, (0, 200, 255), 1)

        # 5. Aks çizgileri (W, L)
        _draw_axes(vis, min_rect, measurements, color_bgr)

        # 6. Diyagonal çizgiler (D1, D2 için CVAI)
        _draw_diagonals(vis, measurements, color_bgr)

    # 7. Ölçüm bilgisi paneli (sağ üst)
    _draw_info_panel(vis, measurements, classification)

    # 8. Şiddet başlık bandı (üst)
    _draw_severity_banner(vis, classification)

    # 9. Güven çubuğu (alt)
    _draw_confidence_bar(vis, confidence)

    # 10. Medikal uyarı (alt)
    _draw_disclaimer(vis)

    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


# ============================================================
# Alt çizim fonksiyonları
# ============================================================

def _overlay_mask(bg: np.ndarray, mask: np.ndarray, color_bgr: tuple, alpha: float):
    """Maskeyi renk ile yarı saydam olarak bindirir."""
    overlay = bg.copy()
    colored = np.zeros_like(bg)
    colored[mask > 0] = color_bgr
    cv2.addWeighted(colored, alpha, bg, 1 - alpha, 0, bg)


def _draw_axes(
    vis: np.ndarray,
    min_rect,
    measurements: dict,
    color_bgr: tuple,
):
    """Kafa genişliği (W) ve uzunluk (L) aks çizgilerini çizer."""
    (cx, cy), (rw, rh), angle = min_rect
    cx, cy = int(cx), int(cy)

    w_px = measurements.get("width_px", 0)
    l_px = measurements.get("length_px", 0)

    # AP eksen açısı
    if rw < rh:
        ap_angle = angle + 90
    else:
        ap_angle = angle

    lat_angle = ap_angle + 90  # Lateral eksen

    def endpoint(center, angle_deg, half_len):
        rad = math.radians(angle_deg)
        return (
            int(center[0] + half_len * math.cos(rad)),
            int(center[1] + half_len * math.sin(rad)),
        )

    # Uzunluk ekseni (L) — mavi
    p1 = endpoint((cx, cy), ap_angle, l_px / 2)
    p2 = endpoint((cx, cy), ap_angle + 180, l_px / 2)
    cv2.line(vis, p1, p2, (255, 100, 0), 2)
    _put_label(vis, f"L={l_px:.0f}px", p1, (255, 100, 0))

    # Genişlik ekseni (W) — yeşil
    p3 = endpoint((cx, cy), lat_angle, w_px / 2)
    p4 = endpoint((cx, cy), lat_angle + 180, w_px / 2)
    cv2.line(vis, p3, p4, (0, 200, 50), 2)
    _put_label(vis, f"W={w_px:.0f}px", p3, (0, 200, 50))

    # Merkez noktası
    cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)


def _draw_diagonals(vis: np.ndarray, measurements: dict, color_bgr: tuple):
    """D1 ve D2 diyagonal çizgilerini çizer (CVAI için)."""
    cx, cy = measurements.get("center", (0, 0))
    cx, cy = int(cx), int(cy)
    d1 = measurements.get("diagonal_d1_px", 0)
    d2 = measurements.get("diagonal_d2_px", 0)
    a1 = measurements.get("diagonal_angle1_deg", 45)
    a2 = measurements.get("diagonal_angle2_deg", -45)

    if d1 > 0:
        rad1 = math.radians(a1)
        ep1a = (int(cx + d1/2 * math.cos(rad1)), int(cy + d1/2 * math.sin(rad1)))
        ep1b = (int(cx - d1/2 * math.cos(rad1)), int(cy - d1/2 * math.sin(rad1)))
        cv2.line(vis, ep1a, ep1b, (200, 0, 200), 1)
        _put_label(vis, f"D1={d1:.0f}", ep1a, (200, 0, 200), scale=0.38)

    if d2 > 0:
        rad2 = math.radians(a2)
        ep2a = (int(cx + d2/2 * math.cos(rad2)), int(cy + d2/2 * math.sin(rad2)))
        ep2b = (int(cx - d2/2 * math.cos(rad2)), int(cy - d2/2 * math.sin(rad2)))
        cv2.line(vis, ep2a, ep2b, (0, 150, 200), 1)
        _put_label(vis, f"D2={d2:.0f}", ep2a, (0, 150, 200), scale=0.38)


def _draw_info_panel(vis: np.ndarray, measurements: dict, classification: dict):
    """Sağ üst köşeye ölçüm değerleri paneli çizer."""
    h, w = vis.shape[:2]

    ci    = measurements.get("cephalic_index", 0)
    cvai  = measurements.get("cvai", 0)
    sym   = measurements.get("symmetry_score", 0)
    circ  = measurements.get("circularity", 0)

    ci_sev   = classification.get("ci_severity", "?")
    cvai_sev = classification.get("cvai_severity", "?")

    lines = [
        f"Sefalik Indeks : {ci:.1f}%",
        f"  ({ci_sev})",
        f"KKAI (CVAI)    : {cvai:.1f}%",
        f"  ({cvai_sev})",
        f"Simetri Skoru  : {sym*100:.0f}%",
        f"Dairesellik    : {circ:.2f}",
    ]

    panel_w = 260
    panel_h = len(lines) * 22 + 16
    x0 = w - panel_w - 10
    y0 = 80  # Başlık bandının altında

    # Yarı saydam arka plan
    overlay = vis.copy()
    cv2.rectangle(overlay, (x0 - 5, y0 - 5), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, vis, 0.28, 0, vis)

    for i, line in enumerate(lines):
        y = y0 + 14 + i * 22
        color = (200, 200, 200) if not line.startswith("  ") else (120, 200, 255)
        cv2.putText(vis, line, (x0, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _draw_severity_banner(vis: np.ndarray, classification: dict):
    """Görüntünün üstüne şiddet derecesini gösteren renkli bant çizer."""
    h, w = vis.shape[:2]
    severity = classification.get("overall", "unknown")
    label    = classification.get("label_tr", severity)
    color    = SEVERITY_COLORS_BGR.get(severity, (100, 100, 100))

    # Üst bant
    cv2.rectangle(vis, (0, 0), (w, 60), color, -1)
    cv2.rectangle(vis, (0, 0), (w, 60), (40, 40, 40), 1)

    txt = f"Sonuc: {label}"
    font_scale = min(0.75, w / 900)
    txt_size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
    tx = (w - txt_size[0]) // 2
    cv2.putText(vis, txt, (tx, 40),
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)


def _draw_confidence_bar(vis: np.ndarray, confidence: float):
    """Alt kısımda güven çubuğu çizer."""
    h, w = vis.shape[:2]
    bar_h = 18
    y = h - bar_h - 28
    bar_w = int(w * 0.6)
    x0 = (w - bar_w) // 2

    cv2.rectangle(vis, (x0, y), (x0 + bar_w, y + bar_h), (40, 40, 40), -1)
    fill_w = int(bar_w * confidence)
    bar_color = _confidence_color(confidence)
    cv2.rectangle(vis, (x0, y), (x0 + fill_w, y + bar_h), bar_color, -1)
    cv2.rectangle(vis, (x0, y), (x0 + bar_w, y + bar_h), (200, 200, 200), 1)

    label = f"Analiz Guveni: {confidence*100:.0f}%"
    cv2.putText(vis, label, (x0 + 4, y + 13),
               cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_disclaimer(vis: np.ndarray):
    """En alta medikal uyarı metni yazar."""
    h, w = vis.shape[:2]
    txt = "UYARI: Bu analiz yalnizca karar destek amacidir. Hekime danismayi ihmal etmeyin."
    scale = min(0.38, w / 1600)
    txt_size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0]
    tx = (w - txt_size[0]) // 2
    cv2.putText(vis, txt, (tx, h - 6),
               cv2.FONT_HERSHEY_SIMPLEX, scale, (80, 80, 255), 1, cv2.LINE_AA)


def _put_label(
    vis: np.ndarray,
    text: str,
    pos: Tuple,
    color: tuple,
    scale: float = 0.42,
):
    """Belirtilen konuma güvenli metin yazar."""
    x, y = int(pos[0]), int(pos[1])
    h, w = vis.shape[:2]
    # Sınır kontrolü
    x = max(4, min(x, w - 120))
    y = max(14, min(y, h - 5))
    cv2.putText(vis, text, (x, y),
               cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _confidence_color(confidence: float) -> tuple:
    """Güven skoru → BGR renk."""
    if confidence >= 0.8:
        return (0, 200, 0)    # yeşil
    if confidence >= 0.6:
        return (0, 200, 200)  # sarı
    if confidence >= 0.4:
        return (0, 120, 255)  # turuncu
    return (0, 0, 200)        # kırmızı


# ============================================================
# Histogram / Şiddet göstergesi grafiği
# ============================================================

def create_gauge_image(classification: dict, width: int = 400, height: int = 120) -> np.ndarray:
    """
    Sefalik indeks değerini gösteren yatay gösterge çubuğu PNG üretir.
    Döner: RGB numpy array
    """
    img = np.ones((height, width, 3), dtype=np.uint8) * 245

    ci = classification.get("scores", {}).get("cephalic_index", 0)

    # Ölçek: 70-110 arası göster
    ci_min, ci_max = 70.0, 110.0
    ci_clipped = max(ci_min, min(ci, ci_max))

    bar_x0, bar_x1 = 40, width - 40
    bar_y0, bar_y1 = 55, 80
    bar_w = bar_x1 - bar_x0

    # Renk geçişi çubuğu
    _draw_gradient_bar(img, bar_x0, bar_y0, bar_x1, bar_y1)

    # Ok işareti (marker)
    marker_x = int(bar_x0 + (ci_clipped - ci_min) / (ci_max - ci_min) * bar_w)
    cv2.circle(img, (marker_x, bar_y0 - 10), 8, (40, 40, 40), -1)
    cv2.line(img, (marker_x, bar_y0 - 2), (marker_x, bar_y1 + 2), (40, 40, 40), 2)

    # SI değeri
    cv2.putText(img, f"SI = {ci:.1f}%", (marker_x - 25, bar_y0 - 22),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)

    # Alt etiketler
    zones = [
        (70, 75,  "Dol."),
        (75, 85,  "Normal"),
        (85, 90,  "Hafif"),
        (90, 95,  "Orta"),
        (95, 110, "Agir"),
    ]
    for z_min, z_max, label in zones:
        zx = int(bar_x0 + (z_min - ci_min) / (ci_max - ci_min) * bar_w)
        cv2.putText(img, label, (zx + 2, bar_y1 + 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 80, 80), 1, cv2.LINE_AA)

    # Alt kenar çizgiler
    for val in [75, 85, 90, 95]:
        vx = int(bar_x0 + (val - ci_min) / (ci_max - ci_min) * bar_w)
        cv2.line(img, (vx, bar_y0), (vx, bar_y1), (80, 80, 80), 1)

    cv2.putText(img, "Sefalik Indeks Gostergesi", (bar_x0, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.50, (60, 60, 60), 1, cv2.LINE_AA)

    return img


def _draw_gradient_bar(img, x0, y0, x1, y1):
    """Yeşilden kırmızıya renk geçişli çubuk çizer (BGR)."""
    w = x1 - x0
    sections = [
        # (start_frac, end_frac, BGR_color)
        (0,    0.125, (255, 165, 0)),   # Dolikosefali – turuncu
        (0.125, 0.375, (0, 180, 0)),    # Normal – yeşil
        (0.375, 0.50,  (0, 200, 220)),  # Hafif – sarı
        (0.50,  0.625, (0, 120, 255)),  # Orta – turuncu
        (0.625, 1.0,   (0, 0, 210)),    # Ağır – kırmızı
    ]
    for frac0, frac1, color in sections:
        xi0 = x0 + int(frac0 * w)
        xi1 = x0 + int(frac1 * w)
        cv2.rectangle(img, (xi0, y0), (xi1, y1), color, -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), (80, 80, 80), 1)
