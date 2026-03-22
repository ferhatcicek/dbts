"""
face_analyzer.py - MediaPipe tabanlı yüz analizi

Frontal (önden) ve lateral (yandan) görüntüler için
MediaPipe Face Mesh kullanarak kafa geometrisi ölçümü.

Üstten görüntü gönderildiğinde bu modül devreye girmez;
yalnızca frontal/lateral analiz modu için çağrılır.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class FaceAnalyzer:
    """MediaPipe Face Mesh sarmalayıcısı."""

    def __init__(self):
        self._mp_face_mesh = None
        self._face_mesh = None
        self._initialized = False
        self._init_mediapipe()

    def _init_mediapipe(self):
        try:
            import mediapipe as mp
            self._mp_face_mesh = mp.solutions.face_mesh
            self._face_mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
            self._initialized = True
            logger.info("MediaPipe Face Mesh başlatıldı.")
        except ImportError:
            logger.warning("MediaPipe kurulu değil; yüz analizi devre dışı.")
        except Exception as e:
            logger.warning(f"MediaPipe başlatma hatası: {e}")

    def analyze(self, img_rgb: np.ndarray, view: str = "front") -> dict:
        """
        Görüntüden yüz/kafa ölçümlerini çıkarır.

        Parametreler:
            img_rgb : RGB görüntü
            view    : 'front' | 'side'

        Döner:
            dict ile ölçümler veya hata bilgisi
        """
        if not self._initialized:
            return {"success": False, "error": "MediaPipe kurulu değil."}

        results = self._face_mesh.process(img_rgb)
        if not results.multi_face_landmarks:
            return {"success": False, "error": "Yüz tespit edilemedi."}

        landmarks = results.multi_face_landmarks[0]
        h, w = img_rgb.shape[:2]

        try:
            if view == "front":
                return self._frontal_analysis(landmarks, h, w)
            elif view == "side":
                return self._lateral_analysis(landmarks, h, w)
            else:
                return {"success": False, "error": f"Bilinmeyen görünüm: {view}"}
        except Exception as e:
            logger.error(f"Yüz analizi hatası: {e}")
            return {"success": False, "error": str(e)}

    def _frontal_analysis(self, landmarks, h: int, w: int) -> dict:
        """
        Önden görünüm: sefalik genişlik, simetri, yüz oranları.

        Kullanılan MediaPipe landmark indeksleri (468-nokta modeli):
          10   → Alın tepesi (vertex)
          152  → Çene ucu (menton)
          234  → Sol temporal bölge
          454  → Sağ temporal bölge
          93   → Sol tragus (kulak)
          323  → Sağ tragus
          1    → Burun ucu (pronasale)
          4    → Philtrum
          33   → Sol göz iç köşesi
          263  → Sağ göz iç köşesi
        """
        def lm(idx):
            lnd = landmarks.landmark[idx]
            return np.array([lnd.x * w, lnd.y * h], dtype=np.float32)

        # ─── Tepe koordinatları ───────────────────────────────
        vertex    = lm(10)    # alın tepesi
        menton    = lm(152)   # çene ucu
        left_tmp  = lm(234)   # sol temporal
        right_tmp = lm(454)   # sağ temporal
        left_ear  = lm(93)    # sol kulak
        right_ear = lm(323)   # sağ kulak
        nasion    = lm(6)     # nazyon (burun kökü)
        nose_tip  = lm(1)     # burun ucu

        # ─── Mesafeler ────────────────────────────────────────
        head_w   = float(np.linalg.norm(right_tmp - left_tmp))
        face_h   = float(np.linalg.norm(menton - vertex))
        ear_w    = float(np.linalg.norm(right_ear - left_ear))
        nose_len = float(np.linalg.norm(nose_tip - nasion))

        # Face index (kranyal genişlik/yüz yüksekliği)
        face_index = (head_w / face_h * 100) if face_h > 0 else 0

        # ─── Simetri: sol-sağ temporal farkı ─────────────────
        mid_x = w / 2
        asymmetry_px = abs((mid_x - left_tmp[0]) - (right_tmp[0] - mid_x))
        asymmetry_pct = (asymmetry_px / head_w * 100) if head_w > 0 else 0

        return {
            "success": True,
            "view": "front",
            "head_width_px":      round(head_w,        1),
            "face_height_px":     round(face_h,        1),
            "ear_width_px":       round(ear_w,         1),
            "nose_length_px":     round(nose_len,      1),
            "face_index":         round(face_index,    2),
            "asymmetry_px":       round(asymmetry_px,  1),
            "asymmetry_pct":      round(asymmetry_pct, 2),
            "landmarks_used": {
                "vertex":     vertex.tolist(),
                "menton":     menton.tolist(),
                "left_temp":  left_tmp.tolist(),
                "right_temp": right_tmp.tolist(),
            },
        }

    def _lateral_analysis(self, landmarks, h: int, w: int) -> dict:
        """
        Yandan görünüm: kafa çıkıntı analizi, alın-çene profili.

        Landmark indeksleri:
          10  → Alın tepesi
          152 → Çene ucu
          1   → Burun ucu
          454 → Sağ temporal (uzakta)
          234 → Sol temporal (görünür taraf)
          4   → Philtrum
          8   → Ağız üstü
        """
        def lm(idx):
            lnd = landmarks.landmark[idx]
            return np.array([lnd.x * w, lnd.y * h], dtype=np.float32)

        vertex    = lm(10)
        menton    = lm(152)
        nose_tip  = lm(1)

        # Alın–çene dikey mesafesi
        head_height = float(np.linalg.norm(menton - vertex))

        # Alnın öne çıkıntısı (vertex'in x pozisyonu referans)
        frontal_prominence = float(abs(vertex[0] - nose_tip[0]))

        # Oksiput çıkıntısı tahmini (arkadaki landmark)
        occipital_est = lm(10)  # yaklaşık referans
        back_pt = lm(13)        # arka kafa tabanı yakın nokta
        occipital_depth = float(abs(back_pt[0] - nose_tip[0]))

        return {
            "success":             True,
            "view":                "side",
            "head_height_px":      round(head_height, 1),
            "frontal_prominence":  round(frontal_prominence, 1),
            "occipital_depth_est": round(occipital_depth, 1),
        }

    def draw_landmarks(
        self,
        img_rgb: np.ndarray,
        analysis_result: dict,
        color: Tuple = (0, 200, 255),
    ) -> np.ndarray:
        """
        Analiz sonucundaki kritik landmark noktalarını görüntüye çizer.
        """
        vis = img_rgb.copy()
        lms = analysis_result.get("landmarks_used", {})
        for name, pt in lms.items():
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(vis, (x, y), 5, color, -1)
            cv2.putText(vis, name[:4], (x + 6, y - 6),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # Kafa genişliği çizgisi
        if "left_temp" in lms and "right_temp" in lms:
            p1 = tuple(int(v) for v in lms["left_temp"])
            p2 = tuple(int(v) for v in lms["right_temp"])
            cv2.line(vis, p1, p2, (0, 255, 0), 2)
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            w_px = analysis_result.get("head_width_px", 0)
            cv2.putText(vis, f"W={w_px:.0f}px", (mid[0] - 30, mid[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        return vis

    def close(self):
        if self._face_mesh:
            self._face_mesh.close()
