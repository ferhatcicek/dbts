"""
app.py - Gradio Web Arayüzü

Deformasyonel Brakisefali Tespit Sistemi
Kullanıcı arayüzü — Çoklu Görünüm Girişi (v2)
"""

import gradio as gr
import numpy as np
import logging
import os
import sys
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessor import load_image_from_numpy, check_image_quality
from src.analyzer     import analyze
from src.config       import APP_TITLE, APP_VERSION, MEDICAL_DISCLAIMER, VIEW_TYPES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# Görünüm rehber diyagramları (SVG + HTML)
# ============================================================

TOP_VIEW_GUIDE_HTML = """
<div style="text-align:center;padding:14px 8px 10px;
            background:linear-gradient(160deg,#ddeeff,#c0d8f5);
            border-radius:12px;border:2px solid #2980b9;margin-bottom:6px;">
  <svg width="130" height="145" viewBox="0 0 130 145" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <radialGradient id="tg1" cx="50%" cy="50%" r="50%">
        <stop offset="0%" stop-color="#fce8d0"/>
        <stop offset="100%" stop-color="#c8845a"/>
      </radialGradient>
    </defs>
    <!-- Camera body at top -->
    <rect x="50" y="2" width="30" height="20" rx="4" fill="#1a3a5c"/>
    <circle cx="65" cy="12" r="7" fill="#5ba3d0"/>
    <circle cx="65" cy="12" r="4" fill="#1a5276"/>
    <rect x="58" y="2" width="6" height="4" rx="1" fill="#0d2137"/>
    <!-- Arrow pointing down from camera -->
    <line x1="65" y1="22" x2="65" y2="33" stroke="#1a3a5c" stroke-width="2.5"/>
    <polygon points="59,33 71,33 65,41" fill="#1a3a5c"/>
    <!-- Head top-view oval -->
    <ellipse cx="65" cy="96" rx="40" ry="50" fill="url(#tg1)" stroke="#7d4e2d" stroke-width="2.5"/>
    <!-- Nose (front of head) -->
    <ellipse cx="65" cy="50" rx="8" ry="6" fill="#b06030" stroke="#7d4e2d" stroke-width="1.5"/>
    <!-- Left ear -->
    <ellipse cx="22" cy="96" rx="6" ry="11" fill="#e8c4a0" stroke="#7d4e2d" stroke-width="1.5"/>
    <!-- Right ear -->
    <ellipse cx="108" cy="96" rx="6" ry="11" fill="#e8c4a0" stroke="#7d4e2d" stroke-width="1.5"/>
    <!-- Width arrow -->
    <line x1="22" y1="134" x2="108" y2="134" stroke="#2471a3" stroke-width="1.5"/>
    <polygon points="22,131 22,137 16,134" fill="#2471a3"/>
    <polygon points="108,131 108,137 114,134" fill="#2471a3"/>
    <text x="65" y="131" text-anchor="middle" font-size="7" fill="#2471a3" font-family="Arial,sans-serif">Genislik (W)</text>
    <!-- Length arrow (left side) -->
    <line x1="8" y1="50" x2="8" y2="142" stroke="#c0392b" stroke-width="1.5"/>
    <polygon points="5,50 11,50 8,44" fill="#c0392b"/>
    <polygon points="5,142 11,142 8,148" fill="#c0392b"/>
    <text x="5" y="99" text-anchor="middle" font-size="7" fill="#c0392b" font-family="Arial,sans-serif"
          transform="rotate(-90,5,99)">Uzunluk (L)</text>
  </svg>
  <p style="margin:5px 0 2px;font-weight:700;color:#1a3a6c;font-size:1.0em;">
    &#128317; Üstten Görünüm</p>
  <p style="margin:0 0 2px;font-size:0.78em;color:#2c5f8a;font-weight:600;">
    Birincil Analiz &nbsp;|&nbsp; SI &amp; KKAI</p>
  <p style="margin:0;font-size:0.72em;color:#555;">Kamerayı kafanın tam üzerinde tut</p>
</div>
"""

FRONT_VIEW_GUIDE_HTML = """
<div style="text-align:center;padding:14px 8px 10px;
            background:linear-gradient(160deg,#ddf5ea,#bce8d2);
            border-radius:12px;border:2px solid #27ae60;margin-bottom:6px;">
  <svg width="130" height="145" viewBox="0 0 130 145" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <radialGradient id="fg1" cx="50%" cy="38%" r="55%">
        <stop offset="0%" stop-color="#fce8d0"/>
        <stop offset="100%" stop-color="#c8845a"/>
      </radialGradient>
    </defs>
    <!-- Camera on right side pointing left -->
    <rect x="100" y="5" width="26" height="18" rx="4" fill="#1a3a5c"/>
    <circle cx="113" cy="14" r="6" fill="#5ba3d0"/>
    <circle cx="113" cy="14" r="3.5" fill="#1a5276"/>
    <line x1="100" y1="14" x2="88" y2="14" stroke="#1a3a5c" stroke-width="2.5"/>
    <polygon points="88,11 88,17 82,14" fill="#1a3a5c"/>
    <!-- Hair -->
    <ellipse cx="65" cy="28" rx="36" ry="18" fill="#4a2c10"/>
    <!-- Head outline -->
    <ellipse cx="65" cy="78" rx="33" ry="54" fill="url(#fg1)" stroke="#7d4e2d" stroke-width="2.2"/>
    <!-- Hair overlay (cover lower edge) -->
    <ellipse cx="65" cy="28" rx="36" ry="14" fill="#4a2c10"/>
    <!-- Ears -->
    <ellipse cx="30" cy="82" rx="5" ry="9" fill="#e8c4a0" stroke="#7d4e2d" stroke-width="1.2"/>
    <ellipse cx="100" cy="82" rx="5" ry="9" fill="#e8c4a0" stroke="#7d4e2d" stroke-width="1.2"/>
    <!-- Left eye -->
    <ellipse cx="48" cy="74" rx="9" ry="6" fill="white" stroke="#555" stroke-width="1.2"/>
    <circle cx="48" cy="74" r="3.5" fill="#2c1a0a"/>
    <circle cx="49.5" cy="72.5" r="1.2" fill="white" opacity="0.7"/>
    <!-- Right eye -->
    <ellipse cx="82" cy="74" rx="9" ry="6" fill="white" stroke="#555" stroke-width="1.2"/>
    <circle cx="82" cy="74" r="3.5" fill="#2c1a0a"/>
    <circle cx="83.5" cy="72.5" r="1.2" fill="white" opacity="0.7"/>
    <!-- Nose -->
    <path d="M 65 82 L 59 95 Q 65 99 71 95 Z" fill="#c07e58" stroke="#906040" stroke-width="1"/>
    <!-- Mouth -->
    <path d="M 52 108 Q 65 118 78 108" fill="none" stroke="#904030" stroke-width="2"/>
    <!-- Symmetry dashed line -->
    <line x1="65" y1="22" x2="65" y2="132" stroke="#c0392b" stroke-width="1.3" stroke-dasharray="5,4"/>
    <text x="65" y="140" text-anchor="middle" font-size="7" fill="#c0392b" font-family="Arial,sans-serif">
      Simetri ekseni</text>
  </svg>
  <p style="margin:5px 0 2px;font-weight:700;color:#145a32;font-size:1.0em;">
    &#128065;&#65039; Önden Görünüm</p>
  <p style="margin:0 0 2px;font-size:0.78em;color:#1e7d46;font-weight:600;">Yüz Simetrisi Analizi</p>
  <p style="margin:0;font-size:0.72em;color:#555;">Bebek kameraya düz baksın</p>
</div>
"""

SIDE_VIEW_GUIDE_HTML = """
<div style="text-align:center;padding:14px 8px 10px;
            background:linear-gradient(160deg,#fff3d6,#ffe0a0);
            border-radius:12px;border:2px solid #e67e22;margin-bottom:6px;">
  <svg width="130" height="145" viewBox="0 0 130 145" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <radialGradient id="sg1" cx="40%" cy="44%" r="56%">
        <stop offset="0%" stop-color="#fce8d0"/>
        <stop offset="100%" stop-color="#c8845a"/>
      </radialGradient>
    </defs>
    <!-- Camera on left side pointing right -->
    <rect x="4" y="5" width="26" height="18" rx="4" fill="#1a3a5c"/>
    <circle cx="17" cy="14" r="6" fill="#5ba3d0"/>
    <circle cx="17" cy="14" r="3.5" fill="#1a5276"/>
    <line x1="30" y1="14" x2="42" y2="14" stroke="#1a3a5c" stroke-width="2.5"/>
    <polygon points="42,11 42,17 48,14" fill="#1a3a5c"/>
    <!-- Head circle (cranium) -->
    <circle cx="75" cy="78" r="46" fill="url(#sg1)" stroke="#7d4e2d" stroke-width="2.2"/>
    <!-- Hair overlay -->
    <path d="M 30 62 Q 38 20 75 16 Q 112 12 120 46 L 120 38
             Q 108 8 75 10 Q 32 12 24 58 Z" fill="#4a2c10"/>
    <!-- Face profile line -->
    <path d="M 50 27 Q 38 43 36 66 Q 34 86 42 103 Q 50 120 65 128"
          fill="none" stroke="#7d4e2d" stroke-width="2"/>
    <!-- Nose bump -->
    <path d="M 38 70 Q 27 76 25 84 Q 28 90 36 88" fill="#c07e58" stroke="#906040" stroke-width="1.5"/>
    <!-- Mouth -->
    <path d="M 34 100 Q 44 112 56 112" fill="none" stroke="#904030" stroke-width="1.8"/>
    <!-- Eye (visible side) -->
    <ellipse cx="52" cy="68" rx="8" ry="5" fill="white" stroke="#555" stroke-width="1.2"/>
    <circle cx="50" cy="68" r="3" fill="#2c1a0a"/>
    <circle cx="51.2" cy="66.8" r="1" fill="white" opacity="0.7"/>
    <!-- Ear (back of head) -->
    <ellipse cx="112" cy="82" rx="7" ry="12" fill="#e8c4a0" stroke="#7d4e2d" stroke-width="1.5"/>
    <ellipse cx="112" cy="82" rx="3.5" ry="6" fill="#c07e58" stroke="none"/>
    <!-- Profile baseline -->
    <line x1="36" y1="130" x2="120" y2="130" stroke="#e67e22" stroke-width="1.3" stroke-dasharray="4,3"/>
    <text x="78" y="140" text-anchor="middle" font-size="7" fill="#c05000" font-family="Arial,sans-serif">
      Profil ekseni</text>
  </svg>
  <p style="margin:5px 0 2px;font-weight:700;color:#7d4e0a;font-size:1.0em;">
    &#128100; Yandan Görünüm</p>
  <p style="margin:0 0 2px;font-size:0.78em;color:#b06010;font-weight:600;">Profil &amp; Oran Analizi</p>
  <p style="margin:0;font-size:0.72em;color:#555;">Kulak tam profilden görünmeli</p>
</div>
"""


# ============================================================
# Yardımcı fonksiyonlar
# ============================================================

def _empty_gauge() -> np.ndarray:
    return np.ones((120, 400, 3), dtype=np.uint8) * 220


def _empty_metrics_html(msg: str = "Analiz bekleniyor...") -> str:
    return f"<p style='color:#888;padding:8px;'>{msg}</p>"


def _placeholder_image(label: str = "Görüntü yüklenmedi") -> np.ndarray:
    img = np.ones((300, 420, 3), dtype=np.uint8) * 215
    cv2.rectangle(img, (10, 10), (409, 289), (180, 180, 180), 2)
    cv2.putText(img, label, (20, 155),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (130, 130, 130), 2, cv2.LINE_AA)
    return img


def _format_metrics_html(measurements: dict, classification: dict) -> str:
    if not measurements:
        return _empty_metrics_html("Ölçüm yapılamadı.")

    ci   = measurements.get("cephalic_index", 0)
    cvai = measurements.get("cvai", 0)
    sym  = measurements.get("symmetry_score", 0) * 100
    circ = measurements.get("circularity", 0)
    w_px = measurements.get("width_px", 0)
    l_px = measurements.get("length_px", 0)
    d1   = measurements.get("diagonal_d1_px", 0)
    d2   = measurements.get("diagonal_d2_px", 0)

    ci_sev   = classification.get("ci_severity",   "?")
    cvai_sev = classification.get("cvai_severity", "?")

    sev_badge = {
        "normal":       '<span style="color:#1a7a3c;font-weight:bold">&#9679; Normal</span>',
        "mild":         '<span style="color:#c8a000;font-weight:bold">&#9679; Hafif</span>',
        "moderate":     '<span style="color:#c85000;font-weight:bold">&#9679; Orta</span>',
        "severe":       '<span style="color:#b00000;font-weight:bold">&#9679; Agir</span>',
        "dolikosefali": '<span style="color:#8a5200;font-weight:bold">&#9679; Dolikosefali</span>',
        "unknown":      '<span style="color:#555;">&#8212;</span>',
    }
    ci_badge   = sev_badge.get(ci_sev,   "&#8212;")
    cvai_badge = sev_badge.get(cvai_sev, "&#8212;")

    rows = []
    if ci > 0:
        rows += [
            f"<tr><td><b>Sefalik Indeks (SI)</b></td>"
            f"<td><b>{ci:.2f}%</b></td><td>{ci_badge}</td></tr>",
            f"<tr><td><b>Krank. Asimetri (KKAI)</b></td>"
            f"<td><b>{cvai:.2f}%</b></td><td>{cvai_badge}</td></tr>",
            f"<tr><td>Simetri Skoru</td><td>{sym:.1f}%</td><td>&#8212;</td></tr>",
            f"<tr><td>Dairesellik</td><td>{circ:.3f}</td><td>&#8212;</td></tr>",
        ]
        if w_px > 0:
            rows += [
                f"<tr><td>Genislik (piksel)</td><td>{w_px:.0f}</td><td>&#8212;</td></tr>",
                f"<tr><td>Uzunluk (piksel)</td><td>{l_px:.0f}</td><td>&#8212;</td></tr>",
            ]
        if d1 > 0:
            rows += [
                f"<tr><td>Diyagonal D1</td><td>{d1:.0f} px</td><td>&#8212;</td></tr>",
                f"<tr><td>Diyagonal D2</td><td>{d2:.0f} px</td><td>&#8212;</td></tr>",
            ]
    else:
        view_m = {k: v for k, v in measurements.items() if isinstance(v, (int, float))}
        for k, v in list(view_m.items())[:10]:
            rows.append(f"<tr><td>{k}</td><td>{v:.3f}</td><td>&#8212;</td></tr>")

    if not rows:
        return _empty_metrics_html("Olcum verisi bulunamadi.")

    return (
        "<style>"
        ".mt{border-collapse:collapse;width:100%;font-size:0.88em;}"
        ".mt th{background:#2471a3;color:#fff;padding:7px 10px;text-align:left;}"
        ".mt td{padding:6px 10px;border-bottom:1px solid #dde;}"
        ".mt tr:nth-child(even){background:#f5f8fc;}"
        ".mt tr:hover{background:#e8f2ff;}"
        "</style>"
        "<table class='mt'>"
        "<tr><th>Olcum</th><th>Deger</th><th>Derece</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def _format_status(result_pairs: list) -> str:
    icons = {
        "normal": "&#9989;", "mild": "&#128993;", "moderate": "&#128992;",
        "severe": "&#128308;", "dolikosefali": "&#128309;", "unknown": "&#10068;",
    }
    parts = []
    for label, result in result_pairs:
        if result is None:
            continue
        if not result.get("success") and result.get("annotated_img") is None:
            parts.append(f"{label}: Hata - {result.get('error', 'Analiz basarisiz')}")
        else:
            cls    = result.get("classification", {})
            sev    = cls.get("overall", "unknown")
            lbl    = cls.get("label_tr", "?")
            conf   = cls.get("confidence", 0)
            method = result.get("seg_result", {}).get("method", "")
            icon   = icons.get(sev, "&#10068;")
            detail = f"Guven: {conf*100:.0f}%"
            if method:
                detail += f" | Seg: {method}"
            parts.append(f"{icon} **{label}:** {lbl} &nbsp;({detail})")
    return "\n\n".join(parts) if parts else "Sonuc alinamadi."


def _blank_outputs():
    """
    Boş/sıfır durum tuple — reset ve başlangıç için.
    14 eleman (build_interface _all_outputs listesiyle eşleşmeli).
    """
    ph = _placeholder_image()
    return (
        gr.update(visible=False),   # top_col
        gr.update(visible=False),   # front_col
        gr.update(visible=False),   # side_col
        "<em style='color:#888;'>Henüz analiz yapılmadı.</em>",   # status
        ph, _empty_gauge(), _empty_metrics_html(),                 # top
        ph, _empty_metrics_html(),                                 # front
        ph, _empty_metrics_html(),                                 # side
        "", None, "",                                              # report, pdf, warning
    )


# ============================================================
# Ana analiz fonksiyonu — çoklu görünüm
# ============================================================

def run_analysis(
    top_image:     np.ndarray,
    front_image:   np.ndarray,
    side_image:    np.ndarray,
    patient_name:  str,
    patient_age:   str,
    patient_gender: str,
    progress=gr.Progress(track_tqdm=True),
) -> tuple:
    """
    3 görünüm için analiz yapar; 14-tuple döndürür.

    Çıktı sırası (_all_outputs ile eşleşmeli):
      0  top_col          gr.update(visible=...)
      1  front_col        gr.update(visible=...)
      2  side_col         gr.update(visible=...)
      3  status_out       Markdown
      4  top_ann          Image
      5  top_gauge        Image
      6  top_metrics      HTML
      7  front_ann        Image
      8  front_metrics    HTML
      9  side_ann         Image
     10  side_metrics     HTML
     11  text_report_out  Textbox
     12  pdf_out          File
     13  quality_warning  Markdown
    """
    images_map = [
        ("top",   "Üstten Görünüm",  top_image),
        ("front", "Önden Görünüm",   front_image),
        ("side",  "Yandan Görünüm",  side_image),
    ]

    submitted = [(vk, vl, img) for vk, vl, img in images_map if img is not None]

    if not submitted:
        outputs = list(_blank_outputs())
        outputs[3] = "&#9888; Lütfen en az bir görüntü yükleyin."
        return tuple(outputs)

    patient_info = {
        "Ad Soyad": patient_name.strip() if patient_name else "",
        "Yasi / Ay": patient_age.strip()  if patient_age  else "",
        "Cinsiyet": patient_gender        if patient_gender else "",
    }

    warnings_list = []
    all_reports   = []
    pdf_path      = None
    result_pairs  = []

    top_ann   = None; top_gauge = None; top_met   = _empty_metrics_html()
    front_ann = None;                   front_met  = _empty_metrics_html()
    side_ann  = None;                   side_met   = _empty_metrics_html()
    has_top = has_front = has_side = False

    total = len(submitted)
    for step_i, (view_key, view_label, img) in enumerate(submitted):
        progress(step_i / total * 0.92,
                 desc=f"{view_label} analiz ediliyor...")
        try:
            img_rgb = load_image_from_numpy(img)
            quality = check_image_quality(img_rgb)
            if quality.get("warnings"):
                warnings_list.extend(quality["warnings"])

            result = analyze(img_rgb, view=view_key, patient_info=patient_info)
            result_pairs.append((view_label, result))

            ann_img = (result.get("annotated_img")
                       if result.get("annotated_img") is not None
                       else img_rgb)

            if result.get("text_report"):
                all_reports.append(
                    f"=== {view_label.upper()} ===\n{result['text_report']}"
                )
            if result.get("pdf_path") and not pdf_path:
                pdf_path = result["pdf_path"]

            mets = result.get("measurements", {})
            cls  = result.get("classification", {})

            if view_key == "top":
                has_top   = True
                top_ann   = ann_img
                top_gauge = result.get("gauge_img") or _empty_gauge()
                top_met   = _format_metrics_html(mets, cls)
            elif view_key == "front":
                has_front = True
                front_ann = ann_img
                front_met = _format_metrics_html(mets, cls)
            elif view_key == "side":
                has_side  = True
                side_ann  = ann_img
                side_met  = _format_metrics_html(mets, cls)

        except Exception as exc:
            logger.error("view=%s hata: %s", view_key, exc, exc_info=True)
            result_pairs.append((view_label, {"success": False, "error": str(exc)}))

    progress(1.0, desc="Tamamlandı!")

    status = _format_status(result_pairs)
    text_report = "\n\n".join(all_reports) if all_reports else "Rapor olusturulamadi."
    warning_txt = "&#9888; " + " | ".join(warnings_list) if warnings_list else ""

    # En az bir sütunu göster
    show_top   = has_top   or (not has_front and not has_side)
    show_front = has_front
    show_side  = has_side

    return (
        gr.update(visible=show_top),
        gr.update(visible=show_front),
        gr.update(visible=show_side),
        status,
        top_ann   if top_ann   is not None else _placeholder_image("Üstten görüntü analiz edilmedi"),
        top_gauge if top_gauge is not None else _empty_gauge(),
        top_met,
        front_ann if front_ann is not None else _placeholder_image("Önden görüntü analiz edilmedi"),
        front_met,
        side_ann  if side_ann  is not None else _placeholder_image("Yandan görüntü analiz edilmedi"),
        side_met,
        text_report,
        pdf_path,
        warning_txt,
    )


# ============================================================
# Statik HTML / Markdown blokları
# ============================================================

HEADER_HTML = f"""
<div style="text-align:center;padding:22px 12px 14px;
            background:linear-gradient(135deg,#1a3a6c,#2980b9);
            border-radius:12px;color:white;margin-bottom:12px;">
  <h1 style="margin:0;font-size:1.9em;letter-spacing:.5px;">&#129504; {APP_TITLE}</h1>
  <p style="margin:8px 0 0;opacity:0.88;font-size:0.95em;">
    Versiyon {APP_VERSION} &nbsp;|&nbsp; Görüntü Tabanlı Klinik Karar Destek Aracı
  </p>
</div>
"""

DISCLAIMER_HTML = f"""
<div style="background:#fff3f3;border:1px solid #c00;border-radius:8px;
            padding:10px 14px;margin-bottom:10px;font-size:0.82em;color:#900;">
  &#9888;&#65039; {MEDICAL_DISCLAIMER}
</div>
"""

HOW_TO_USE_HTML = """
<div style="font-family:Arial,sans-serif;max-width:960px;margin:0 auto;padding:10px 18px;">

<!-- ══════════════════════════════════════════════════════ -->
<h2 style="color:#1a3a6c;border-bottom:3px solid #2980b9;padding-bottom:8px;">
  &#128247; 1. Fotoğraf Çekim Rehberi
</h2>

<div style="display:flex;flex-wrap:wrap;gap:16px;margin-bottom:24px;">

  <!-- Üstten -->
  <div style="flex:1;min-width:220px;background:#ddeeff;border:2px solid #2980b9;
              border-radius:10px;padding:14px;">
    <h3 style="color:#1a3a6c;margin:0 0 8px;">&#128317; Üstten Görünüm</h3>
    <p style="font-size:.85em;color:#333;margin:0 0 8px;">
      <b>Birincil analiz:</b> Sefalik İndeks (SI) ve Kranyal Asimetri (KKAI) ölçümü yapılır.
    </p>
    <ul style="font-size:.83em;color:#333;margin:0;padding-left:18px;">
      <li>Bebeği <b>sırt üstü yatırın</b> ya da dik oturtun</li>
      <li>Kamerayı kafanın tam <b>üzerinde</b> tutun (kuşbakışı)</li>
      <li>Kafanın <b>tamamı çerçevede</b> olsun, kenarlar kesilmesin</li>
      <li>Düz, sade arka plan; <b>gölgesiz aydınlatma</b></li>
      <li>Saçlar mümkünse <b>toplanmış/kısa</b> olsun</li>
    </ul>
  </div>

  <!-- Önden -->
  <div style="flex:1;min-width:220px;background:#ddf5ea;border:2px solid #27ae60;
              border-radius:10px;padding:14px;">
    <h3 style="color:#145a32;margin:0 0 8px;">&#128065;&#65039; Önden Görünüm</h3>
    <p style="font-size:.85em;color:#333;margin:0 0 8px;">
      <b>İkincil analiz:</b> MediaPipe ile yüz simetri analizi yapılır.
    </p>
    <ul style="font-size:.83em;color:#333;margin:0;padding-left:18px;">
      <li>Bebek kameraya <b>düz baksın</b></li>
      <li>Yüzü <b>ortala</b>, sağa-sola eğme</li>
      <li>Her iki kulak eşit görünebilmeli</li>
      <li>Göz hizası <b>yatay</b> olsun</li>
    </ul>
  </div>

  <!-- Yandan -->
  <div style="flex:1;min-width:220px;background:#fff3d6;border:2px solid #e67e22;
              border-radius:10px;padding:14px;">
    <h3 style="color:#7d4e0a;margin:0 0 8px;">&#128100; Yandan Görünüm</h3>
    <p style="font-size:.85em;color:#333;margin:0 0 8px;">
      <b>Profil analizi:</b> Ön-arka oran değerlendirmesi.
    </p>
    <ul style="font-size:.83em;color:#333;margin:0;padding-left:18px;">
      <li>Kulak <b>tam profilden</b> görünmeli</li>
      <li>Baş <b>dik</b> tutulsun, öne-arkaya eğilmesin</li>
      <li>Saç ve bone <b>kafa sınırını kapatmasın</b></li>
    </ul>
  </div>
</div>


<!-- ══════════════════════════════════════════════════════ -->
<h2 style="color:#1a3a6c;border-bottom:3px solid #2980b9;padding-bottom:8px;">
  &#128200; 2. Sefalik İndeks (SI) Hesabı
</h2>

<div style="display:flex;flex-wrap:wrap;gap:24px;align-items:flex-start;margin-bottom:24px;">

  <!-- SVG diagram -->
  <div style="flex:0 0 auto;text-align:center;">
    <svg width="230" height="220" viewBox="0 0 230 220" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <radialGradient id="hg1" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="#fce8d0"/>
          <stop offset="100%" stop-color="#d4906a"/>
        </radialGradient>
      </defs>
      <!-- Head oval -->
      <ellipse cx="115" cy="108" rx="72" ry="90" fill="url(#hg1)" stroke="#7d4e2d" stroke-width="3"/>
      <!-- Nose -->
      <ellipse cx="115" cy="22" rx="13" ry="9" fill="#b06030" stroke="#7d4e2d" stroke-width="2"/>
      <!-- Left ear -->
      <ellipse cx="37" cy="108" rx="9" ry="16" fill="#e8c4a0" stroke="#7d4e2d" stroke-width="2"/>
      <!-- Right ear -->
      <ellipse cx="193" cy="108" rx="9" ry="16" fill="#e8c4a0" stroke="#7d4e2d" stroke-width="2"/>

      <!-- Width (W) arrow - horizontal -->
      <line x1="37" y1="108" x2="193" y2="108" stroke="#2471a3" stroke-width="2.5"/>
      <polygon points="37,104 37,112 28,108" fill="#2471a3"/>
      <polygon points="193,104 193,112 202,108" fill="#2471a3"/>
      <text x="115" y="104" text-anchor="middle" font-size="13" fill="#2471a3"
            font-family="Arial,sans-serif" font-weight="bold">W (Genişlik)</text>

      <!-- Length (L) arrow - vertical -->
      <line x1="115" y1="22" x2="115" y2="196" stroke="#c0392b" stroke-width="2.5"/>
      <polygon points="111,22 119,22 115,13" fill="#c0392b"/>
      <polygon points="111,196 119,196 115,205" fill="#c0392b"/>
      <text x="132" y="116" text-anchor="start" font-size="13" fill="#c0392b"
            font-family="Arial,sans-serif" font-weight="bold"
            transform="rotate(90,132,110)">L (Uzunluk)</text>

      <!-- minAreaRect dashed box -->
      <rect x="43" y="18" width="144" height="182" rx="2"
            fill="none" stroke="#888" stroke-width="1.5" stroke-dasharray="6,4"/>
      <text x="115" y="212" text-anchor="middle" font-size="10" fill="#555"
            font-family="Arial,sans-serif">Minimum Alan Dikdörtgeni</text>
    </svg>
    <p style="font-size:.78em;color:#555;margin:4px 0 0;">
      <code>cv2.minAreaRect()</code> ile W ve L bulunur
    </p>
  </div>

  <!-- Açıklama -->
  <div style="flex:1;min-width:260px;">
    <div style="background:#eaf4ff;border-left:5px solid #2980b9;padding:14px 16px;
                border-radius:6px;margin-bottom:14px;">
      <p style="margin:0 0 6px;font-size:.95em;color:#1a3a6c;font-weight:bold;">Formül</p>
      <p style="margin:0;font-size:1.4em;color:#1a3a6c;font-family:Georgia,serif;">
        SI &nbsp;=&nbsp; (W &divide; L) &times; 100
      </p>
      <p style="margin:6px 0 0;font-size:.82em;color:#555;">
        W = kafa genişliği (piksel) &nbsp;|&nbsp; L = kafa uzunluğu (piksel)
      </p>
    </div>

    <p style="font-size:.88em;color:#333;margin:0 0 10px;">
      Sefalik İndeks, kafanın <b>ön-arka uzunluğuna oranla ne kadar geniş</b> olduğunu ölçer.
      SI arttıkça kafa daha <b>yuvarlak/yassı</b>, azaldıkça daha <b>uzun/dar</b> görünür.
    </p>

    <table style="width:100%;border-collapse:collapse;font-size:.84em;">
      <tr style="background:#2471a3;color:#fff;">
        <th style="padding:6px 10px;text-align:left;">SI Aralığı</th>
        <th style="padding:6px 10px;text-align:left;">Sınıflandırma</th>
        <th style="padding:6px 10px;text-align:left;">Renk</th>
      </tr>
      <tr style="background:#f0f8ff;">
        <td style="padding:5px 10px;">SI &lt; 75%</td>
        <td style="padding:5px 10px;">Dolikosefali (uzun/dar kafa)</td>
        <td style="padding:5px 10px;"><span style="color:#8a5200;font-weight:bold;">&#9679; Kahve</span></td>
      </tr>
      <tr>
        <td style="padding:5px 10px;">75% – 84%</td>
        <td style="padding:5px 10px;"><b>Normal</b></td>
        <td style="padding:5px 10px;"><span style="color:#1a7a3c;font-weight:bold;">&#9679; Yeşil</span></td>
      </tr>
      <tr style="background:#f0f8ff;">
        <td style="padding:5px 10px;">85% – 89%</td>
        <td style="padding:5px 10px;">Hafif Brakisefali</td>
        <td style="padding:5px 10px;"><span style="color:#c8a000;font-weight:bold;">&#9679; Sarı</span></td>
      </tr>
      <tr>
        <td style="padding:5px 10px;">90% – 95%</td>
        <td style="padding:5px 10px;">Orta Şiddetli Brakisefali</td>
        <td style="padding:5px 10px;"><span style="color:#c85000;font-weight:bold;">&#9679; Turuncu</span></td>
      </tr>
      <tr style="background:#fff0f0;">
        <td style="padding:5px 10px;">SI &gt; 95%</td>
        <td style="padding:5px 10px;"><b>Ağır Brakisefali</b></td>
        <td style="padding:5px 10px;"><span style="color:#b00000;font-weight:bold;">&#9679; Kırmızı</span></td>
      </tr>
    </table>

    <p style="font-size:.78em;color:#666;margin:8px 0 0;">
      Kaynak: Argenta LC ve ark., <em>J Craniofac Surg</em>, 2004
    </p>
  </div>
</div>


<!-- ══════════════════════════════════════════════════════ -->
<h2 style="color:#1a3a6c;border-bottom:3px solid #2980b9;padding-bottom:8px;">
  &#128200; 3. Kranyal Kasa Asimetri İndeksi (KKAI / CVAI) Hesabı
</h2>

<div style="display:flex;flex-wrap:wrap;gap:24px;align-items:flex-start;margin-bottom:24px;">

  <!-- SVG -->
  <div style="flex:0 0 auto;text-align:center;">
    <svg width="250" height="230" viewBox="0 0 250 230" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <radialGradient id="hg2" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="#fce8d0"/>
          <stop offset="100%" stop-color="#d4906a"/>
        </radialGradient>
      </defs>
      <!-- Asymmetric head (plagiocephaly simulation) -->
      <!-- Slightly rotated / flattened on one side -->
      <ellipse cx="122" cy="115" rx="72" ry="88"
               fill="url(#hg2)" stroke="#7d4e2d" stroke-width="3"
               transform="rotate(8,122,115)"/>
      <!-- Nose -->
      <ellipse cx="118" cy="28" rx="12" ry="8" fill="#b06030" stroke="#7d4e2d" stroke-width="1.5"/>

      <!-- AP axis (front-back) dashed center line -->
      <line x1="118" y1="18" x2="126" y2="210" stroke="#888" stroke-width="1.5"
            stroke-dasharray="6,4"/>

      <!-- D1: 45° diagonal (sol-ön → sağ-arka) - LONGER -->
      <line x1="60" y1="52" x2="188" y2="185"
            stroke="#2471a3" stroke-width="2.5"/>
      <polygon points="60,48 56,56 64,56" fill="#2471a3"/>
      <polygon points="184,181 192,181 188,189" fill="#2471a3"/>
      <text x="100" y="93" font-size="13" fill="#2471a3" font-weight="bold"
            font-family="Arial,sans-serif" transform="rotate(45,100,93)">D1</text>

      <!-- D2: 135° diagonal (sağ-ön → sol-arka) - SHORTER -->
      <line x1="178" y1="55" x2="66" y2="178"
            stroke="#c0392b" stroke-width="2.5"/>
      <polygon points="178,51 174,59 182,57" fill="#c0392b"/>
      <polygon points="62,174 70,174 66,182" fill="#c0392b"/>
      <text x="155" y="95" font-size="13" fill="#c0392b" font-weight="bold"
            font-family="Arial,sans-serif" transform="rotate(-45,155,95)">D2</text>

      <!-- Labels -->
      <text x="125" y="222" text-anchor="middle" font-size="10" fill="#555"
            font-family="Arial,sans-serif">
        D1 &gt; D2 → Asimetri var (Plagiasefali)
      </text>
      <text x="23" y="50" font-size="10" fill="#2471a3" font-family="Arial,sans-serif">45°</text>
      <text x="188" y="50" font-size="10" fill="#c0392b" font-family="Arial,sans-serif">135°</text>
    </svg>
    <p style="font-size:.78em;color:#555;margin:4px 0 0;">
      AP eksenine göre ±45° diyagonaller
    </p>
  </div>

  <!-- Açıklama -->
  <div style="flex:1;min-width:260px;">
    <div style="background:#fff5e6;border-left:5px solid #e67e22;padding:14px 16px;
                border-radius:6px;margin-bottom:14px;">
      <p style="margin:0 0 6px;font-size:.95em;color:#7d4e0a;font-weight:bold;">Formül</p>
      <p style="margin:0;font-size:1.35em;color:#7d4e0a;font-family:Georgia,serif;">
        KKAI &nbsp;=&nbsp; |D1 &minus; D2| &divide; max(D1, D2) &times; 100
      </p>
      <p style="margin:6px 0 0;font-size:.82em;color:#555;">
        D1 = 45° diyagonal uzunluğu &nbsp;|&nbsp; D2 = 135° diyagonal uzunluğu
      </p>
    </div>

    <p style="font-size:.88em;color:#333;margin:0 0 10px;">
      KKAI, kafanın <b>ön-arka (AP) eksenine göre 45° ve 135° açısındaki</b> iki diyagonalin
      farkını ölçer. Değer sıfıra yakınsa simetri yüksek, büyükse kafa asimetrisi (plagiosefali)
      var demektir.
    </p>

    <div style="background:#f0fff4;border:1px solid #27ae60;border-radius:6px;
                padding:10px 14px;margin-bottom:10px;font-size:.84em;">
      <b>Nasıl hesaplanır?</b>
      <ol style="margin:6px 0 0;padding-left:20px;color:#333;">
        <li>Kafanın ağırlık merkezi (centroid) bulunur</li>
        <li>Kontur noktaları üzerine <b>AP ekseni</b> yerleştirilir</li>
        <li>AP ekseninden ±45° açıyla iki ayrı diyagonal çizilir</li>
        <li>Her diyagonal boyunca konturu aşan uzunluk (D1 ve D2) ölçülür</li>
        <li>Formül uygulanarak KKAI hesaplanır</li>
      </ol>
    </div>

    <table style="width:100%;border-collapse:collapse;font-size:.84em;">
      <tr style="background:#c8760a;color:#fff;">
        <th style="padding:6px 10px;text-align:left;">KKAI Aralığı</th>
        <th style="padding:6px 10px;text-align:left;">Şiddet</th>
      </tr>
      <tr style="background:#f0f8ff;">
        <td style="padding:5px 10px;">KKAI &lt; 3.5%</td>
        <td style="padding:5px 10px;"><span style="color:#1a7a3c;font-weight:bold;">&#9679; Normal</span></td>
      </tr>
      <tr>
        <td style="padding:5px 10px;">3.5% – 6.25%</td>
        <td style="padding:5px 10px;"><span style="color:#c8a000;font-weight:bold;">&#9679; Hafif Plagiosefali</span></td>
      </tr>
      <tr style="background:#f0f8ff;">
        <td style="padding:5px 10px;">6.25% – 8.75%</td>
        <td style="padding:5px 10px;"><span style="color:#c85000;font-weight:bold;">&#9679; Orta Plagiosefali</span></td>
      </tr>
      <tr style="background:#fff0f0;">
        <td style="padding:5px 10px;">KKAI &gt; 8.75%</td>
        <td style="padding:5px 10px;"><span style="color:#b00000;font-weight:bold;">&#9679; Ağır Plagiosefali</span></td>
      </tr>
    </table>
    <p style="font-size:.78em;color:#666;margin:8px 0 0;">
      Kaynak: Ott R ve ark., <em>Neuropediatrics</em>, 2007
    </p>
  </div>
</div>


<!-- ══════════════════════════════════════════════════════ -->
<h2 style="color:#1a3a6c;border-bottom:3px solid #2980b9;padding-bottom:8px;">
  &#128200; 4. Simetri Skoru ve Dairesellik
</h2>

<div style="display:flex;flex-wrap:wrap;gap:24px;align-items:flex-start;margin-bottom:24px;">

  <!-- SVG Simetri -->
  <div style="flex:0 0 auto;text-align:center;">
    <svg width="230" height="190" viewBox="0 0 230 190" xmlns="http://www.w3.org/2000/svg">
      <!-- Sol yarı (mavi) -->
      <ellipse cx="115" cy="100" rx="65" ry="80"
               fill="#3498db" opacity="0.35" stroke="none"/>
      <!-- Sağ yarı biraz kayık (kırmızı) -->
      <ellipse cx="128" cy="106" rx="60" ry="74"
               fill="#e74c3c" opacity="0.35" stroke="none"/>
      <!-- Kesişim (mor) -->
      <ellipse cx="120" cy="103" rx="55" ry="70"
               fill="#8e44ad" opacity="0.25" stroke="none"/>
      <!-- Orta eksen -->
      <line x1="115" y1="5" x2="115" y2="185"
            stroke="#333" stroke-width="2" stroke-dasharray="7,5"/>
      <!-- Etiketler -->
      <text x="72" y="98" text-anchor="middle" font-size="12" fill="#1a5276"
            font-weight="bold" font-family="Arial,sans-serif">Sol</text>
      <text x="154" y="98" text-anchor="middle" font-size="12" fill="#922b21"
            font-weight="bold" font-family="Arial,sans-serif">Sağ</text>
      <text x="115" y="182" text-anchor="middle" font-size="10" fill="#555"
            font-family="Arial,sans-serif">Kesişim (IoU)</text>
      <!-- Brace annotation -->
      <text x="115" y="18" text-anchor="middle" font-size="10" fill="#555"
            font-family="Arial,sans-serif">Dikey simetri ekseni</text>
    </svg>
    <p style="font-size:.78em;color:#555;margin:4px 0 0;">IoU (Kesişim/Birleşim) tabanlı simetri</p>
  </div>

  <div style="flex:1;min-width:260px;">
    <div style="background:#f0fff4;border-left:5px solid #27ae60;padding:14px 16px;
                border-radius:6px;margin-bottom:12px;">
      <p style="margin:0 0 4px;font-size:.95em;color:#145a32;font-weight:bold;">Simetri Skoru</p>
      <p style="margin:0;font-size:1.25em;color:#145a32;font-family:Georgia,serif;">
        Simetri &nbsp;=&nbsp; Kesişim &divide; Birleşim &nbsp;(IoU)
      </p>
      <p style="margin:6px 0 0;font-size:.82em;color:#555;">
        Maske dikey eksende yansıtılır; orijinal ve yansıma örtüşmesi ölçülür.
        1.0 = mükemmel simetri &nbsp;|&nbsp; 0.0 = tam asimetri
      </p>
    </div>

    <div style="background:#fef9e7;border-left:5px solid #f1c40f;padding:14px 16px;
                border-radius:6px;">
      <p style="margin:0 0 4px;font-size:.95em;color:#7d6608;font-weight:bold;">Dairesellik</p>
      <p style="margin:0;font-size:1.25em;color:#7d6608;font-family:Georgia,serif;">
        C &nbsp;=&nbsp; 4&pi;A &divide; P&sup2;
      </p>
      <p style="margin:6px 0 0;font-size:.82em;color:#555;">
        A = kontur alanı &nbsp;|&nbsp; P = kontur çevresi.<br/>
        Değer 1.0'a yakınsa kafa <b>daha yuvarlak</b>.
        Değer düştükçe şekil <b>daha uzun/düzensiz</b>.
      </p>
    </div>
  </div>
</div>


<!-- ══════════════════════════════════════════════════════ -->
<h2 style="color:#1a3a6c;border-bottom:3px solid #2980b9;padding-bottom:8px;">
  &#9881;&#65039; 5. Segmentasyon Aşamaları
</h2>

<div style="margin-bottom:24px;">
  <p style="font-size:.88em;color:#333;margin:0 0 12px;">
    Sistem, kafayı arka plandan ayırmak için <b>4 kademeli yedek algoritma</b> kullanır.
    Birinci yöntem başarısız olursa otomatik olarak sonrakine geçilir:
  </p>
  <div style="display:flex;flex-wrap:wrap;gap:10px;">
    <div style="flex:1;min-width:140px;background:linear-gradient(135deg,#1a5276,#2980b9);
                color:#fff;padding:12px 14px;border-radius:8px;text-align:center;">
      <div style="font-size:1.6em;">&#129504;</div>
      <div style="font-weight:bold;font-size:.9em;margin:4px 0;">1. rembg / U-Net AI</div>
      <div style="font-size:.75em;opacity:.9;">Derin öğrenme tabanlı, en yüksek güven</div>
    </div>
    <div style="flex:0 0 auto;display:flex;align-items:center;color:#888;font-size:1.2em;">&#8594;</div>
    <div style="flex:1;min-width:140px;background:linear-gradient(135deg,#1e8449,#27ae60);
                color:#fff;padding:12px 14px;border-radius:8px;text-align:center;">
      <div style="font-size:1.6em;">&#9986;&#65039;</div>
      <div style="font-weight:bold;font-size:.9em;margin:4px 0;">2. GrabCut</div>
      <div style="font-size:.75em;opacity:.9;">OpenCV interaktif segmentasyon</div>
    </div>
    <div style="flex:0 0 auto;display:flex;align-items:center;color:#888;font-size:1.2em;">&#8594;</div>
    <div style="flex:1;min-width:140px;background:linear-gradient(135deg,#9a7d0a,#d4ac0d);
                color:#fff;padding:12px 14px;border-radius:8px;text-align:center;">
      <div style="font-size:1.6em;">&#127769;</div>
      <div style="font-weight:bold;font-size:.9em;margin:4px 0;">3. Otsu Eşikleme</div>
      <div style="font-size:.75em;opacity:.9;">Histogram tabanlı global eşikleme</div>
    </div>
    <div style="flex:0 0 auto;display:flex;align-items:center;color:#888;font-size:1.2em;">&#8594;</div>
    <div style="flex:1;min-width:140px;background:linear-gradient(135deg,#922b21,#e74c3c);
                color:#fff;padding:12px 14px;border-radius:8px;text-align:center;">
      <div style="font-size:1.6em;">&#128269;</div>
      <div style="font-weight:bold;font-size:.9em;margin:4px 0;">4. Uyarlamalı</div>
      <div style="font-size:.75em;opacity:.9;">Yerel eşikleme (son çare)</div>
    </div>
  </div>
</div>


<!-- ══════════════════════════════════════════════════════ -->
<h2 style="color:#1a3a6c;border-bottom:3px solid #2980b9;padding-bottom:8px;">
  &#9888;&#65039; 6. Önemli Sınırlılıklar
</h2>
<div style="background:#fff3f3;border:1px solid #e74c3c;border-radius:8px;
            padding:14px 18px;font-size:.86em;color:#7b241c;margin-bottom:16px;">
  <ul style="margin:0;padding-left:20px;line-height:1.8;">
    <li>Bu sistem <b>piksel tabanlı oran</b> ölçümü yapar; gerçek fiziksel boyut ölçmez.</li>
    <li>Görüntü kalitesi (odak, aydınlatma, açı) sonucu <b>doğrudan</b> etkiler.</li>
    <li><b>Saç kalınlığı</b>, <b>perspektif bozulması</b> ve <b>kıyafet/bone</b> ölçüm hatasına yol açabilir.</li>
    <li>Segmentasyon yöntemi sonuç güven skorunu etkiler — rembg en güvenilir yöntemdir.</li>
    <li>Bu araç yalnızca <b>klinik karar destek</b> amaçlıdır; <b>tıbbi tanı yerine geçmez</b>.</li>
    <li>Kesin tanı için mutlaka <b>kraniyofasiyal uzman</b> veya <b>çocuk nöroloji hekimi</b> ile görüşün.</li>
  </ul>
</div>

</div>
"""

ABOUT_MD = f"""
## Deformasyonel Brakisefali Hakkında

**Brakisefali** (Yunanca: brakhy = kısa, kephale = kafa), kafanın arka kısmının yassılaşmasıyla
ortaya çıkan bir baş şekli bozukluğudur.

### Sefalik İndeks (SI) Referans Değerleri
| Aralık | Sınıf |
|---|---|
| SI < 75% | Dolikosefali (uzun/dar kafa) |
| 75% – 84% | **Normal** |
| 85% – 89% | Hafif Brakisefali |
| 90% – 95% | Orta Şiddetli Brakisefali |
| SI > 95% | **Ağır Brakisefali** |

### Kranyal Kasa Asimetri İndeksi (KKAI / CVAI)
| Aralık | Şiddet |
|---|---|
| KKAI < 3.5% | Normal |
| 3.5% – 6.25% | Hafif |
| 6.25% – 8.75% | Orta |
| KKAI > 8.75% | Ağır |

### Referanslar
- Argenta LC ve ark. *J Craniofac Surg*, 2004
- Ott R ve ark. *Neuropediatrics*, 2007
- Loveday BP, de Chalain TB. *J Craniofac Surg*, 2001

---
**{APP_TITLE}** v{APP_VERSION}
"""


# ============================================================
# Gradio arayüzü
# ============================================================

def build_interface() -> gr.Blocks:

    with gr.Blocks(title=APP_TITLE) as demo:

        gr.HTML(HEADER_HTML)
        gr.HTML(DISCLAIMER_HTML)

        with gr.Tabs():

            # ── Tab 1: Analiz ────────────────────────────────────────
            with gr.Tab("Analiz"):

                # ────────────────────────────────────────────────────
                # VERİ GİRİŞİ — üstte
                # ────────────────────────────────────────────────────
                gr.Markdown(
                    "## Görüntü Yükleme\n"
                    "Analiz etmek istediğiniz görünüm(ler)e ait "
                    "fotoğrafları aşağıdaki alanlara yükleyin. "
                    "Birden fazla görünüm yüklenirse hepsi analiz edilir."
                )

                with gr.Row(equal_height=False):

                    with gr.Column(scale=1, min_width=200):
                        gr.HTML(TOP_VIEW_GUIDE_HTML)
                        top_image_in = gr.Image(
                            label="Üstten Görünüm Fotoğrafı",
                            type="numpy",
                            sources=["upload", "clipboard"],
                            height=240,
                        )

                    with gr.Column(scale=1, min_width=200):
                        gr.HTML(FRONT_VIEW_GUIDE_HTML)
                        front_image_in = gr.Image(
                            label="Önden Görünüm Fotoğrafı",
                            type="numpy",
                            sources=["upload", "clipboard"],
                            height=240,
                        )

                    with gr.Column(scale=1, min_width=200):
                        gr.HTML(SIDE_VIEW_GUIDE_HTML)
                        side_image_in = gr.Image(
                            label="Yandan Görünüm Fotoğrafı",
                            type="numpy",
                            sources=["upload", "clipboard"],
                            height=240,
                        )

                gr.Markdown("### Hasta Bilgileri (İsteğe Bağlı)")
                with gr.Row():
                    patient_name   = gr.Textbox(
                        label="Ad Soyad", placeholder="—", scale=2)
                    patient_age    = gr.Textbox(
                        label="Yaş / Ay", placeholder="ör. 6 ay", scale=1)
                    patient_gender = gr.Dropdown(
                        label="Cinsiyet",
                        choices=["—", "Erkek", "Kız"],
                        value="—",
                        scale=1,
                    )

                analyze_btn     = gr.Button("Analiz Et", variant="primary", size="lg")
                quality_warning = gr.Markdown()

                gr.HTML(
                    "<hr style='margin:24px 0 20px;"
                    "border:none;border-top:2px solid #cbd5e0;'/>"
                )

                # ────────────────────────────────────────────────────
                # SONUÇLAR — altta
                # ────────────────────────────────────────────────────
                gr.Markdown("## Analiz Sonuçları")

                status_out = gr.Markdown(
                    value=(
                        "<em style='color:#999;'>"
                        "Henüz analiz yapılmadı. "
                        "Görüntü yükleyip 'Analiz Et' butonuna basın."
                        "</em>"
                    )
                )

                # ── Üstten görünüm sonuç sütunu (başlangıçta gizli)
                with gr.Column(visible=False) as top_col:
                    gr.HTML(
                        "<div style='border-top:3px solid #2980b9;"
                        "margin:12px 0 10px;'></div>"
                        "<h3 style='color:#1a3a6c;margin:0 0 8px;'>"
                        "&#128317; Üstten Görünüm — SI &amp; KKAI Sonuçları</h3>"
                    )
                    with gr.Row():
                        with gr.Column(scale=3):
                            top_ann_out = gr.Image(
                                label="Anotasyonlu Görüntü",
                                type="numpy",
                                height=370,
                            )
                        with gr.Column(scale=2):
                            top_gauge_out = gr.Image(
                                label="Sefalik İndeks Göstergesi",
                                type="numpy",
                                height=145,
                            )
                            top_metrics_out = gr.HTML(
                                value=_empty_metrics_html()
                            )

                # ── Önden görünüm sonuç sütunu (başlangıçta gizli)
                with gr.Column(visible=False) as front_col:
                    gr.HTML(
                        "<div style='border-top:3px solid #27ae60;"
                        "margin:12px 0 10px;'></div>"
                        "<h3 style='color:#145a32;margin:0 0 8px;'>"
                        "&#128065;&#65039; Önden Görünüm — Yüz Simetrisi Sonuçları</h3>"
                    )
                    with gr.Row():
                        with gr.Column(scale=3):
                            front_ann_out = gr.Image(
                                label="Anotasyonlu Görüntü",
                                type="numpy",
                                height=370,
                            )
                        with gr.Column(scale=2):
                            front_metrics_out = gr.HTML(
                                value=_empty_metrics_html()
                            )

                # ── Yandan görünüm sonuç sütunu (başlangıçta gizli)
                with gr.Column(visible=False) as side_col:
                    gr.HTML(
                        "<div style='border-top:3px solid #e67e22;"
                        "margin:12px 0 10px;'></div>"
                        "<h3 style='color:#7d4e0a;margin:0 0 8px;'>"
                        "&#128100; Yandan Görünüm — Profil Sonuçları</h3>"
                    )
                    with gr.Row():
                        with gr.Column(scale=3):
                            side_ann_out = gr.Image(
                                label="Anotasyonlu Görüntü",
                                type="numpy",
                                height=370,
                            )
                        with gr.Column(scale=2):
                            side_metrics_out = gr.HTML(
                                value=_empty_metrics_html()
                            )

                # ── Raporlar
                with gr.Accordion("Metin Raporu", open=False):
                    text_report_out = gr.Textbox(
                        label="Rapor",
                        lines=18,
                        max_lines=30,
                        interactive=False,
                    )
                pdf_out = gr.File(label="PDF Raporu İndir")

            # ── Tab 2: Kullanım Kılavuzu ──────────────────────────────
            with gr.Tab("Kullanım Kılavuzu"):
                gr.HTML(HOW_TO_USE_HTML)

            # ── Tab 3: Hakkında ───────────────────────────────────────
            with gr.Tab("Hakkında"):
                gr.Markdown(ABOUT_MD)

        # ── Ortak çıktı listesi ───────────────────────────────────────
        _all_outputs = [
            top_col, front_col, side_col,
            status_out,
            top_ann_out, top_gauge_out, top_metrics_out,
            front_ann_out, front_metrics_out,
            side_ann_out, side_metrics_out,
            text_report_out, pdf_out,
            quality_warning,
        ]

        # ── Analiz butonu ─────────────────────────────────────────────
        analyze_btn.click(
            fn=run_analysis,
            inputs=[
                top_image_in, front_image_in, side_image_in,
                patient_name, patient_age, patient_gender,
            ],
            outputs=_all_outputs,
            show_progress="full",
        )

        # ── Görüntü değişince sonuçları sıfırla ──────────────────────
        for img_comp in [top_image_in, front_image_in, side_image_in]:
            img_comp.change(
                fn=_blank_outputs,
                inputs=[],
                outputs=_all_outputs,
            )

    return demo


# ============================================================
# Giriş noktası
# ============================================================

if __name__ == "__main__":
    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    )
