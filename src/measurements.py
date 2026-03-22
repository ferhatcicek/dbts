"""
measurements.py - Kafatası ölçüm algoritmaları

Deformasyonel Brakisefali için temel ölçümler:

  1. Sefalik İndeks (SI / Cephalic Index)
       SI = (Kafa Genişliği / Kafa Uzunluğu) × 100
       Kaynak: Argenta sınıflaması, WHO standartları

  2. Kranyal Kasa Asimetri İndeksi (KKAI / CVAI)
       KKAI = |D1 - D2| / D2 × 100
       D1, D2: AP eksenine 45° açılı diyagonal ölçümler
       Kaynak: Ott ve ark. 2007; Loveday & de Chalain 2001

  3. Ek geometrik metrikler (ovallik, simetri, dairesellik)

Tüm fonksiyonlar OpenCV kontur formatı kabul eder.
"""

import cv2
import numpy as np
import math
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# ============================================================
# Ana ölçüm fonksiyonu
# ============================================================

def compute_all_measurements(contour: np.ndarray, mask: np.ndarray) -> dict:
    """
    Kafa konturundan tüm tanısal ölçümleri hesaplar.

    Parametreler:
        contour : kafa kontur dizisi (cv2 formatı)
        mask    : binary maske

    Döner:
        dict — tüm ölçüm değerleri
    """
    if contour is None:
        return _empty_measurements()

    results = {}

    # --- Temel şekil özellikleri ---
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    results["area_px"] = float(area)
    results["perimeter_px"] = float(perimeter)
    results["circularity"] = _calc_circularity(area, perimeter)

    # --- Minimum alanlı döndürülmüş dikdörtgen (minAreaRect) ---
    min_rect = cv2.minAreaRect(contour)
    (cx, cy), (rect_w, rect_h), angle = min_rect
    results["center"] = (float(cx), float(cy))
    results["rect_angle"] = float(angle)

    # W = genişlik (kısa kenar), L = uzunluk (uzun kenar)
    rect_w, rect_h = float(rect_w), float(rect_h)
    if rect_w > rect_h:
        w_px, l_px = rect_h, rect_w
    else:
        w_px, l_px = rect_w, rect_h

    results["width_px"] = w_px
    results["length_px"] = l_px

    # --- Sefalik İndeks ---
    ci = _calc_cephalic_index(w_px, l_px)
    results["cephalic_index"] = round(ci, 2)

    # --- Elips uydurma ---
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        (ex, ey), (ema, emi), eangle = ellipse
        results["ellipse"] = {
            "center": (float(ex), float(ey)),
            "major_axis": float(max(ema, emi)),
            "minor_axis": float(min(ema, emi)),
            "angle": float(eangle),
        }
        # Elipse dayalı SI (doğrulama amaçlı)
        ellipse_ci = (min(ema, emi) / max(ema, emi)) * 100 if max(ema, emi) > 0 else 0
        results["ellipse_ci"] = round(float(ellipse_ci), 2)
    else:
        results["ellipse"] = None
        results["ellipse_ci"] = ci

    # --- CVAI (diyagonal ölçümler) ---
    cvai_data = _calc_cvai(contour, min_rect)
    results.update(cvai_data)

    # --- Simetri skoru ---
    results["symmetry_score"] = _calc_symmetry(mask, (int(cx), int(cy)))

    # --- Konvekslık ---
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    results["convexity"] = float(area / hull_area) if hull_area > 0 else 0.0

    # --- Aspect ratio ---
    x_b, y_b, w_b, h_b = cv2.boundingRect(contour)
    results["bounding_box"] = (int(x_b), int(y_b), int(w_b), int(h_b))
    results["aspect_ratio"] = float(w_b) / h_b if h_b > 0 else 0.0

    return results


# ============================================================
# Bireysel ölçüm fonksiyonları
# ============================================================

def _calc_cephalic_index(width: float, length: float) -> float:
    """
    Sefalik İndeks = (Genişlik / Uzunluk) × 100
    Uzunluk sıfırsa 0 döner.
    """
    if length <= 0:
        return 0.0
    return (width / length) * 100.0


def _calc_circularity(area: float, perimeter: float) -> float:
    """
    Dairesellik = 4π × Alan / Çevre²
    Mükemmel çember → 1.0
    """
    if perimeter <= 0:
        return 0.0
    return float((4 * math.pi * area) / (perimeter ** 2))


def _calc_cvai(contour: np.ndarray, min_rect: tuple) -> dict:
    """
    Kranyal Kasa Asimetri İndeksi (CVAI):
      D1 = AP eksenine +45° açılı diyagonal uzunluk (kontur üzerinden)
      D2 = AP eksenine -45° açılı diyagonal uzunluk
      CVAI = |D1 - D2| / max(D1, D2) × 100

    Döner: D1, D2, CVAI değerleri içeren dict
    """
    try:
        (cx, cy), (rw, rh), angle = min_rect

        # AP ekseninin açısı — minAreaRect açısına göre belirlenir
        # OpenCV minAreaRect: açı -90..0 arası, uzun aks için düzeltme
        long_side = max(rw, rh)
        short_side = min(rw, rh)

        if rw < rh:
            # Dikdörtgenin uzun kenarı dikey → AP eksen açısı = angle + 90
            ap_angle_deg = angle + 90.0
        else:
            ap_angle_deg = angle

        # Diyagonal açılar (45° sağ ve 45° sol)
        diag1_angle = math.radians(ap_angle_deg + 45)
        diag2_angle = math.radians(ap_angle_deg - 45)

        d1 = _diagonal_length_through_center(contour, cx, cy, diag1_angle)
        d2 = _diagonal_length_through_center(contour, cx, cy, diag2_angle)

        if d1 == 0 or d2 == 0:
            return {"diagonal_d1_px": 0, "diagonal_d2_px": 0, "cvai": 0.0}

        # Büyük değer her zaman D2 (payda)
        d_max = max(d1, d2)
        d_min = min(d1, d2)
        cvai = abs(d1 - d2) / d_max * 100.0

        return {
            "diagonal_d1_px": round(d1, 1),
            "diagonal_d2_px": round(d2, 1),
            "cvai": round(cvai, 2),
            "diagonal_angle1_deg": round(math.degrees(diag1_angle), 1),
            "diagonal_angle2_deg": round(math.degrees(diag2_angle), 1),
        }
    except Exception as e:
        logger.warning(f"CVAI hesaplama hatası: {e}")
        return {"diagonal_d1_px": 0, "diagonal_d2_px": 0, "cvai": 0.0}


def _diagonal_length_through_center(
    contour: np.ndarray, cx: float, cy: float, angle_rad: float
) -> float:
    """
    Merkez (cx, cy) üzerinden verilen açıda geçen çizginin
    konturla kesişim noktaları arasındaki mesafeyi hesaplar.
    """
    # Kontur noktalarına projeksiyon
    pts = contour.reshape(-1, 2).astype(np.float32)

    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)

    # Her nokta için çizgi üzerindeki projeksiyon skaları
    projections = (pts[:, 0] - cx) * dx + (pts[:, 1] - cy) * dy

    pos_proj = projections[projections >= 0]
    neg_proj = projections[projections < 0]

    if len(pos_proj) == 0 or len(neg_proj) == 0:
        return 0.0

    return float(pos_proj.max() + abs(neg_proj.min()))


def _calc_symmetry(mask: np.ndarray, center: Tuple[int, int]) -> float:
    """
    Simetri skoru: maskenin merkeze göre yatay simetrisini ölçer.
    Döner: 0 (tamamen asimetrik) – 1 (mükemmel simetrik)
    """
    cx = center[0]
    left = mask[:, :cx]
    right = mask[:, cx:]

    # Sol ve sağı aynı genişliğe getir
    min_w = min(left.shape[1], right.shape[1])
    if min_w == 0:
        return 0.0

    left_crop = left[:, -min_w:]
    right_crop = right[:, :min_w]
    right_flip = np.fliplr(right_crop)

    # Örtüşme oranı
    intersection = np.logical_and(left_crop > 0, right_flip > 0).sum()
    union = np.logical_or(left_crop > 0, right_flip > 0).sum()

    if union == 0:
        return 0.0
    return float(intersection / union)


def _empty_measurements() -> dict:
    """Kontur yokken boş sonuç döner."""
    return {
        "area_px": 0,
        "perimeter_px": 0,
        "circularity": 0,
        "center": (0, 0),
        "rect_angle": 0,
        "width_px": 0,
        "length_px": 0,
        "cephalic_index": 0,
        "ellipse": None,
        "ellipse_ci": 0,
        "diagonal_d1_px": 0,
        "diagonal_d2_px": 0,
        "cvai": 0,
        "symmetry_score": 0,
        "convexity": 0,
        "bounding_box": (0, 0, 0, 0),
        "aspect_ratio": 0,
    }


# ============================================================
# MediaPipe tabanlı ölçümler (Frontal / Lateral görünüm)
# ============================================================

def compute_frontal_measurements(landmarks, img_shape: Tuple[int, int]) -> dict:
    """
    MediaPipe Face Mesh landmark'larından yüz/kafa ölçümleri çıkarır.
    Özellikle frontal (önden) görüntüler için.

    Landmark indeksleri (MediaPipe 468-nokta modeli):
      - Temporal nokta sol  : 234
      - Temporal nokta sağ  : 454
      - Glabella (alın üst) : 10
      - Mentum (çene alt)   : 152
      - Sol kulak           : 93
      - Sağ kulak           : 323
    """
    h, w = img_shape[:2]

    def lm(idx):
        lnd = landmarks.landmark[idx]
        return np.array([lnd.x * w, lnd.y * h])

    try:
        # Kafa genişliği (temporal–temporal)
        left_temp  = lm(234)
        right_temp = lm(454)
        head_width_px = float(np.linalg.norm(right_temp - left_temp))

        # Yüz yüksekliği (glabella – çene)
        glabella = lm(10)
        mentum   = lm(152)
        face_height_px = float(np.linalg.norm(mentum - glabella))

        # Kulak noktaları (yüz gözlemi amaçlı)
        left_ear  = lm(93)
        right_ear = lm(323)
        ear_width = float(np.linalg.norm(right_ear - left_ear))

        # Yüz Genişlik/Yükseklik oranı
        face_ratio = head_width_px / face_height_px if face_height_px > 0 else 0

        return {
            "head_width_px": round(head_width_px, 1),
            "face_height_px": round(face_height_px, 1),
            "ear_width_px": round(ear_width, 1),
            "face_ratio": round(face_ratio, 3),
            "left_temporal_pt": left_temp.tolist(),
            "right_temporal_pt": right_temp.tolist(),
            "view": "frontal",
        }
    except Exception as e:
        logger.warning(f"Frontal ölçüm hatası: {e}")
        return {"view": "frontal", "error": str(e)}
