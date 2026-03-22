"""
preprocessor.py - Görüntü ön işleme modülü

Yüklenen görüntüyü analiz için hazırlar:
  - Boyut normalizasyonu
  - Renk uzayı dönüşümü
  - Kalite kontrolü
  - EXIF rotasyon düzeltme
"""

import cv2
import numpy as np
from PIL import Image, ExifTags
import io
import logging

from src.config import TARGET_IMAGE_SIZE

logger = logging.getLogger(__name__)


def load_image_from_bytes(file_bytes: bytes) -> np.ndarray:
    """
    Byte dizisinden görüntü yükler (Gradio yükleme ile uyumlu).
    EXIF rotasyonunu düzeltir.
    Dönen görüntü: RGB, uint8
    """
    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
        pil_img = _fix_exif_rotation(pil_img)
        pil_img = pil_img.convert("RGB")
        return np.array(pil_img, dtype=np.uint8)
    except Exception as e:
        logger.error(f"Görüntü yükleme hatası: {e}")
        raise ValueError(f"Görüntü yüklenemedi: {e}")


def load_image_from_path(path: str) -> np.ndarray:
    """
    Dosya yolundan görüntü yükler.
    Dönen görüntü: RGB, uint8
    """
    try:
        pil_img = Image.open(path)
        pil_img = _fix_exif_rotation(pil_img)
        pil_img = pil_img.convert("RGB")
        return np.array(pil_img, dtype=np.uint8)
    except Exception as e:
        logger.error(f"Dosya yükleme hatası: {e}")
        raise ValueError(f"Dosya yüklenemedi: {e}")


def load_image_from_numpy(arr: np.ndarray) -> np.ndarray:
    """
    Gradio'nun numpy array olarak verdiği görüntüyü normalize eder.
    Dönen görüntü: RGB, uint8
    """
    if arr is None:
        raise ValueError("Görüntü boş (None).")
    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    return arr


def preprocess(img_rgb: np.ndarray, max_size: int = TARGET_IMAGE_SIZE) -> dict:
    """
    Ana ön işleme fonksiyonu.

    Parametreler:
        img_rgb   : RGB numpy array
        max_size  : Maksimum kenar uzunluğu (piksel)

    Dönen değer:
        dict içinde:
          - 'original'    : Orijinal boyut RGB görüntü
          - 'resized'     : Ölçeklendirilmiş RGB görüntü
          - 'gray'        : Gri tonlamalı görüntü
          - 'scale'       : Ölçek faktörü (orijinal → resized)
          - 'shape'       : (h, w) resized boyutu
    """
    if not isinstance(img_rgb, np.ndarray):
        raise TypeError("Görüntü numpy array olmalıdır.")
    if img_rgb.ndim not in (2, 3):
        raise ValueError("Görüntü 2D veya 3D array olmalıdır.")

    original = img_rgb.copy()
    h, w = img_rgb.shape[:2]

    # Boyut normalizasyonu
    scale = 1.0
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized = img_rgb.copy()

    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

    return {
        "original": original,
        "resized": resized,
        "gray": gray,
        "scale": scale,
        "shape": resized.shape[:2],
    }


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """
    CLAHE ile kontrast artırımı — segmentasyon kalitesini iyileştirir.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def check_image_quality(img_rgb: np.ndarray) -> dict:
    """
    Görüntü kalite kontrolü yapar.

    Döner:
        dict: {'ok': bool, 'warnings': list[str]}
    """
    warnings = []
    h, w = img_rgb.shape[:2]

    # Çok küçük görüntü
    if min(h, w) < 200:
        warnings.append(f"Görüntü çok küçük ({w}×{h}px). En az 400×400 px önerilir.")

    # Çok karanlık görüntü
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mean_brightness = float(gray.mean())
    if mean_brightness < 40:
        warnings.append("Görüntü çok karanlık. Daha iyi aydınlatma kullanın.")
    elif mean_brightness > 220:
        warnings.append("Görüntü çok açık (aşırı pozlama). Daha az aydınlatma kullanın.")

    # Bulanıklık tespiti (Laplacian varyansı)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if lap_var < 50:
        warnings.append("Görüntü bulanık görünüyor. Daha net bir fotoğraf çekin.")

    return {"ok": len(warnings) == 0, "warnings": warnings, "brightness": mean_brightness, "sharpness": lap_var}


# ──────────────────────────────────────────────────────────────
# Yardımcı fonksiyonlar
# ──────────────────────────────────────────────────────────────

def _fix_exif_rotation(pil_img: Image.Image) -> Image.Image:
    """EXIF verilerine göre fotoğraf rotasyonunu düzeltir."""
    try:
        exif = pil_img._getexif()
        if exif is None:
            return pil_img
        orientation_key = next(
            (k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None
        )
        if orientation_key is None:
            return pil_img
        orientation = exif.get(orientation_key)
        rotations = {3: 180, 6: 270, 8: 90}
        if orientation in rotations:
            pil_img = pil_img.rotate(rotations[orientation], expand=True)
    except Exception:
        pass
    return pil_img
