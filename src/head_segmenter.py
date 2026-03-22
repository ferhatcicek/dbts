"""
head_segmenter.py - Kafatası bölgesi tespiti ve segmentasyonu

Tepe (üstten) görüntülerde kafa bölgesini izole eder.
Birden fazla yöntem deneyen kademeli (fallback) mimari:
  1. rembg (AI tabanlı, en güvenilir)
  2. GrabCut (OpenCV, otomatik merkez başlatma)
  3. Otsu eşikleme + morfolojik işlemler
  4. Uyarlamalı eşikleme (son çare)
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple

from src.config import (
    GRABCUT_ITERATIONS,
    MIN_HEAD_AREA_RATIO,
    MAX_HEAD_AREA_RATIO,
    MIN_CIRCULARITY,
)

logger = logging.getLogger(__name__)


# ============================================================
# Ana segmentasyon fonksiyonu
# ============================================================

def segment_head(img_rgb: np.ndarray) -> dict:
    """
    Görüntüdeki kafa bölgesini tespit eder ve maskesini çıkarır.

    Parametreler:
        img_rgb : RGB numpy array

    Döner:
        dict:
          - 'mask'      : binary maske (255=kafa, 0=arka plan)
          - 'contour'   : kafa konturu (cv2 contour formatı)
          - 'bbox'      : (x, y, w, h) bounding box
          - 'method'    : kullanılan yöntem adı
          - 'confidence': tahmini güven skoru (0-1)
          - 'success'   : bool
    """
    h, w = img_rgb.shape[:2]
    total_area = h * w

    # Yöntem 1: rembg (AI tabanlı arka plan kaldırma)
    result = _try_rembg(img_rgb)
    if result and _validate_mask(result["mask"], total_area):
        logger.info(f"Segmentasyon: rembg yöntemi başarılı (güven: {result['confidence']:.2f})")
        return _finalize_result(result, img_rgb)

    # Yöntem 2: GrabCut
    result = _try_grabcut(img_rgb)
    if result and _validate_mask(result["mask"], total_area):
        logger.info(f"Segmentasyon: GrabCut yöntemi başarılı (güven: {result['confidence']:.2f})")
        return _finalize_result(result, img_rgb)

    # Yöntem 3: Otsu eşikleme
    result = _try_otsu(img_rgb)
    if result and _validate_mask(result["mask"], total_area):
        logger.info(f"Segmentasyon: Otsu yöntemi başarılı (güven: {result['confidence']:.2f})")
        return _finalize_result(result, img_rgb)

    # Yöntem 4: Uyarlamalı eşikleme (son çare)
    result = _try_adaptive(img_rgb)
    logger.warning("Segmentasyon: Uyarlamalı eşikleme kullanıldı (düşük güven).")
    return _finalize_result(result, img_rgb)


# ============================================================
# Segmentasyon yöntemleri
# ============================================================

def _try_rembg(img_rgb: np.ndarray) -> Optional[dict]:
    """AI tabanlı arka plan kaldırma (rembg kütüphanesi)."""
    try:
        from rembg import remove
        from PIL import Image
        import io

        pil_in = Image.fromarray(img_rgb)
        pil_out = remove(pil_in)  # RGBA çıktı
        out_arr = np.array(pil_out)

        # Alpha kanalından maske çıkar
        alpha = out_arr[:, :, 3]
        mask = (alpha > 128).astype(np.uint8) * 255

        # Morfolojik temizlik
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        return {"mask": mask, "method": "rembg", "confidence": 0.90}
    except ImportError:
        logger.debug("rembg yüklü değil, atlanıyor.")
        return None
    except Exception as e:
        logger.debug(f"rembg hatası: {e}")
        return None


def _try_grabcut(img_rgb: np.ndarray) -> Optional[dict]:
    """GrabCut algoritması — merkezi kafa olarak varsayar."""
    try:
        h, w = img_rgb.shape[:2]
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Merkez bölgeyi dikdörtgen olarak al (kenarlarda arka plan varsayımı)
        margin_x = max(int(w * 0.08), 10)
        margin_y = max(int(h * 0.08), 10)
        rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

        mask_gc = np.zeros((h, w), dtype=np.uint8)
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)

        cv2.grabCut(img_bgr, mask_gc, rect, bgd_model, fgd_model,
                    GRABCUT_ITERATIONS, cv2.GC_INIT_WITH_RECT)

        # GrabCut mask: 0,2 = arka plan; 1,3 = ön plan
        fg_mask = np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD),
                           255, 0).astype(np.uint8)

        # Morfolojik işlemler
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Sadece en büyük bileşeni tut
        fg_mask = _keep_largest_component(fg_mask)

        return {"mask": fg_mask, "method": "grabcut", "confidence": 0.75}
    except Exception as e:
        logger.debug(f"GrabCut hatası: {e}")
        return None


def _try_otsu(img_rgb: np.ndarray) -> Optional[dict]:
    """Otsu eşikleme — düzgün arka planlı görüntüler için."""
    try:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # CLAHE ile kontrast artır
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)

        # Gaussian bulanıklık
        blurred = cv2.GaussianBlur(gray_eq, (11, 11), 0)

        # Otsu eşikleme
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Arka planın rengi daha koyu mu yoksa aydınlık mı? Otomatik belirle.
        h, w = gray.shape
        corners = [gray[0, 0], gray[0, w-1], gray[h-1, 0], gray[h-1, w-1]]
        corner_mean = np.mean(corners)
        center_val = gray[h//2, w//2]

        # Köşeler daha koyu → kafa açık → ters çevir gerekebilir
        if corner_mean > center_val:
            thresh = cv2.bitwise_not(thresh)

        # Morfolojik işlemler
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=4)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = _keep_largest_component(mask)

        return {"mask": mask, "method": "otsu", "confidence": 0.60}
    except Exception as e:
        logger.debug(f"Otsu hatası: {e}")
        return None


def _try_adaptive(img_rgb: np.ndarray) -> dict:
    """Uyarlamalı eşikleme — son çare yöntemi."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 51, -5
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = _keep_largest_component(mask)

    return {"mask": mask, "method": "adaptive", "confidence": 0.40}


# ============================================================
# Yardımcı fonksiyonlar
# ============================================================

def _validate_mask(mask: np.ndarray, total_area: int) -> bool:
    """Maskenin makul bir kafa bölgesi oluşturup oluşturmadığını kontrol eder."""
    fg_area = np.count_nonzero(mask)
    ratio = fg_area / total_area

    if ratio < MIN_HEAD_AREA_RATIO:
        logger.debug(f"Maske çok küçük (oran={ratio:.3f})")
        return False
    if ratio > MAX_HEAD_AREA_RATIO:
        logger.debug(f"Maske çok büyük (oran={ratio:.3f})")
        return False

    # Kontur var mı?
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    # En büyük konturun yuvarlıklığını kontrol et
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return False
    circularity = (4 * np.pi * area) / (perimeter ** 2)

    if circularity < MIN_CIRCULARITY:
        logger.debug(f"Kontur yeterince yuvarlak değil (circularity={circularity:.3f})")
        return False

    return True


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Maskede sadece en büyük bağlı bileşeni korur."""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return mask

    # 0 = arka plan, skip
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    clean_mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    return clean_mask


def _finalize_result(result: dict, img_rgb: np.ndarray) -> dict:
    """Maske + görüntüden kontür ve bbox hesaplar, sonucu tamamlar."""
    mask = result["mask"]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {
            "mask": mask,
            "contour": None,
            "bbox": None,
            "method": result.get("method", "unknown"),
            "confidence": 0.0,
            "success": False,
        }

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)

    return {
        "mask": mask,
        "contour": contour,
        "bbox": (x, y, w, h),
        "method": result.get("method", "unknown"),
        "confidence": result.get("confidence", 0.5),
        "success": True,
    }


def apply_mask_to_image(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Maskeyi görüntüye uygulayarak kafa dışını siler (beyaz arka plan)."""
    result = img_rgb.copy()
    result[mask == 0] = 255  # Arka plan beyaz
    return result
