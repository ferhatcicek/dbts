"""
reporter.py - PDF / HTML rapor oluşturucu

Analiz sonuçlarından klinik değerlendirme raporu üretir.
fpdf2 kütüphanesi kullanılır (hafif, bağımlılık az).
rembg / reportlab yoksa düz metin rapor fallback'i devreye girer.
"""

import io
import os
import logging
import tempfile
from datetime import datetime
from typing import Optional

import numpy as np
from PIL import Image

from src.config import (
    APP_TITLE,
    APP_VERSION,
    MEDICAL_DISCLAIMER,
    SEVERITY_LABELS_TR,
)
from src.classifier import get_ci_interpretation, get_cvai_interpretation

logger = logging.getLogger(__name__)


# ============================================================
# Ana rapor fonksiyonu
# ============================================================

def generate_pdf_report(
    img_rgb: np.ndarray,
    annotated_rgb: np.ndarray,
    measurements: dict,
    classification: dict,
    patient_info: Optional[dict] = None,
) -> Optional[str]:
    """
    PDF raporu oluşturur ve geçici dosyaya kaydeder.

    Döner: PDF dosyasının yolu (str) veya None (hata durumunda)
    """
    try:
        from fpdf import FPDF, XPos, YPos
        return _build_pdf(img_rgb, annotated_rgb, measurements, classification, patient_info)
    except ImportError:
        logger.warning("fpdf2 kurulu değil; HTML rapor deneniyor.")
        return generate_html_report(measurements, classification, patient_info)
    except Exception as e:
        logger.error(f"PDF oluşturma hatası: {e}")
        return None


def generate_html_report(
    measurements: dict,
    classification: dict,
    patient_info: Optional[dict] = None,
) -> Optional[str]:
    """
    Basit HTML raporu oluşturur ve geçici dosyaya kaydeder.
    """
    try:
        html = _build_html(measurements, classification, patient_info)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False,
            encoding="utf-8", prefix="brakisefali_rapor_"
        ) as f:
            f.write(html)
            return f.name
    except Exception as e:
        logger.error(f"HTML rapor oluşturma hatası: {e}")
        return None


def generate_text_report(
    measurements: dict,
    classification: dict,
    patient_info: Optional[dict] = None,
) -> str:
    """
    Yedek metin raporu döner (her zaman kullanılabilir).
    """
    lines = []
    now = datetime.now().strftime("%d.%m.%Y %H:%M")

    lines.append("=" * 60)
    lines.append(f"  {APP_TITLE}")
    lines.append(f"  Versiyon: {APP_VERSION}")
    lines.append(f"  Analiz Tarihi: {now}")
    lines.append("=" * 60)

    if patient_info:
        lines.append("\n--- HASTA BİLGİLERİ ---")
        for k, v in patient_info.items():
            if v:
                lines.append(f"  {k}: {v}")

    lines.append("\n--- ÖLÇÜM SONUÇLARI ---")
    ci    = measurements.get("cephalic_index", 0)
    cvai  = measurements.get("cvai", 0)
    sym   = measurements.get("symmetry_score", 0)
    circ  = measurements.get("circularity", 0)
    w_px  = measurements.get("width_px", 0)
    l_px  = measurements.get("length_px", 0)

    lines.append(f"  Sefalik İndeks (SI)         : {ci:.2f}%")
    lines.append(f"    {get_ci_interpretation(ci)}")
    lines.append(f"  Krank. Kasa Asimetri İnd.   : {cvai:.2f}%")
    lines.append(f"    {get_cvai_interpretation(cvai)}")
    lines.append(f"  Simetri Skoru               : {sym*100:.1f}%")
    lines.append(f"  Dairesellik                 : {circ:.3f}")
    lines.append(f"  Kafa Genişliği (piksel)     : {w_px:.0f}")
    lines.append(f"  Kafa Uzunluğu (piksel)      : {l_px:.0f}")

    lines.append("\n--- SINIFLANDIRMA ---")
    overall = classification.get("overall", "?")
    label   = classification.get("label_tr", overall)
    conf    = classification.get("confidence", 0)
    lines.append(f"  Genel Sonuç    : {label}")
    lines.append(f"  SI Derecesi    : {SEVERITY_LABELS_TR.get(classification.get('ci_severity', '?'), '?')}")
    lines.append(f"  KKAI Derecesi  : {SEVERITY_LABELS_TR.get(classification.get('cvai_severity', '?'), '?')}")
    lines.append(f"  Analiz Güveni  : {conf*100:.0f}%")

    lines.append("\n--- ÖNERİ ---")
    rec = classification.get("recommendation", "")
    for chunk in [rec[i:i+70] for i in range(0, len(rec), 70)]:
        lines.append(f"  {chunk}")

    lines.append("\n" + "=" * 60)
    lines.append("YASAL UYARI:")
    for chunk in [MEDICAL_DISCLAIMER[i:i+58] for i in range(0, len(MEDICAL_DISCLAIMER), 58)]:
        lines.append(f"  {chunk}")
    lines.append("=" * 60)

    return "\n".join(lines)


# ============================================================
# PDF inşa eden iç fonksiyon
# ============================================================

def _tr_to_ascii(text: str) -> str:
    """Türkçe + emoji karakterleri ASCII karşılıklarıyla değiştirir (fpdf Helvetica için)."""
    replacements = {
        "Ş": "S", "ş": "s", "Ğ": "G", "ğ": "g",
        "Ü": "U", "ü": "u", "Ö": "O", "ö": "o",
        "İ": "I", "ı": "i", "Ç": "C", "ç": "c",
        "→": "->", "←": "<-", "↑": "^", "↓": "v",
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "⚠️": "[!]", "⚠": "[!]", "✅": "[OK]", "✓": "[OK]",
        "🔴": "[!!!]", "🟡": "[!]", "🟠": "[!!]", "🔵": "[*]",
        "❌": "[X]", "❔": "[?]", "📷": "[img]", "📄": "[doc]",
        "ℹ️": "[i]", "🔬": "[~]", "📊": "[=]", "📖": "[?]",
        "👶": "", "🧠": "", "📂": "", "📥": "", "📋": "",
        "⚕": "", "🏥": "",
    }
    for char, repl in replacements.items():
        text = text.replace(char, repl)
    # Kalan Latin-1 dışı karakterleri kaldır
    text = text.encode("latin-1", errors="replace").decode("latin-1")
    return text


def _build_pdf(img_rgb, annotated_rgb, measurements, classification, patient_info) -> str:
    from fpdf import FPDF, XPos, YPos

    pdf = FPDF()
    pdf.add_page()

    # Unicode font desteği için fallback — fpdf2 Helvetica Latin-1 only
    # DejaVu font dosyası varsa kullan, yoksa Helvetica + transliterasyon
    FONT = "Helvetica"
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        font_path = os.path.join(base_dir, "..", "assets", "DejaVuSans.ttf")
        font_bold = os.path.join(base_dir, "..", "assets", "DejaVuSans-Bold.ttf")
        if os.path.exists(font_path):
            pdf.add_font("DejaVu", "", font_path, uni=True)
            if os.path.exists(font_bold):
                pdf.add_font("DejaVu", "B", font_bold, uni=True)
            FONT = "DejaVu"
    except Exception:
        pass

    now = datetime.now().strftime("%d.%m.%Y %H:%M")

    # Türkçe dönüştürme sarmalayıcısı — yalnızca Helvetica için aktif
    def t(text: str) -> str:
        return _tr_to_ascii(text) if FONT == "Helvetica" else text

    # ── Başlık
    pdf.set_fill_color(41, 128, 185)
    pdf.rect(0, 0, 210, 25, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font(FONT, "B", 14)
    pdf.cell(0, 12, t(APP_TITLE), align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font(FONT, "", 9)
    pdf.cell(0, 8, f"Analiz Tarihi: {now}  |  Versiyon: {APP_VERSION}",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    # ── Hasta bilgileri
    if patient_info:
        pdf.set_font(FONT, "B", 10)
        pdf.cell(0, 7, t("HASTA BILGILERI"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font(FONT, "", 9)
        for k, v in patient_info.items():
            if v:
                pdf.cell(0, 6, t(f"  {k}: {v}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(3)

    # ── Annotasyon görüntüsü
    try:
        pil_ann = Image.fromarray(annotated_rgb)
        tf_path = tempfile.mktemp(suffix=".jpg", prefix="ann_")
        pil_ann.save(tf_path, "JPEG", quality=88)
        pdf.image(tf_path, x=10, w=90)
        try:
            os.unlink(tf_path)
        except Exception:
            pass
        pdf.ln(2)
    except Exception as e:
        logger.warning(f"PDF görüntü eklenemedi: {e}")

    # ── Ölçüm tablosu
    pdf.set_font(FONT, "B", 10)
    pdf.cell(0, 7, t("OLCUM SONUCLARI"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font(FONT, "", 9)

    ci    = measurements.get("cephalic_index", 0)
    cvai  = measurements.get("cvai", 0)
    sym   = measurements.get("symmetry_score", 0)
    circ  = measurements.get("circularity", 0)

    rows = [
        (t("Sefalik Indeks (SI)"),      f"{ci:.2f}%",    t(get_ci_interpretation(ci))),
        (t("Krank. Kasa Asim. (KKAI)"), f"{cvai:.2f}%",  t(get_cvai_interpretation(cvai))),
        (t("Simetri Skoru"),            f"{sym*100:.1f}%", ""),
        (t("Dairesellik"),              f"{circ:.3f}",    ""),
    ]

    pdf.set_fill_color(230, 240, 255)
    for i, (name, val, interp) in enumerate(rows):
        fill = i % 2 == 0
        pdf.set_fill_color(235, 245, 255) if fill else pdf.set_fill_color(255, 255, 255)
        pdf.cell(70, 7, name, border=1, fill=fill)
        pdf.cell(30, 7, val, border=1, fill=fill)
        pdf.cell(0, 7, interp, border=1, fill=fill, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(4)

    # ── Sınıflandırma
    overall = classification.get("overall", "?")
    label   = classification.get("label_tr", overall)
    conf    = classification.get("confidence", 0)
    rec     = classification.get("recommendation", "")

    severity_colors = {
        "normal":       (0, 150, 50),
        "mild":         (180, 130, 0),
        "moderate":     (200, 80, 0),
        "severe":       (180, 0, 0),
        "dolikosefali": (150, 80, 0),
    }
    sc = severity_colors.get(overall, (80, 80, 80))

    pdf.set_font(FONT, "B", 10)
    pdf.cell(0, 7, t("SINIFLANDIRMA"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font(FONT, "", 9)
    pdf.set_text_color(*sc)
    pdf.cell(0, 8, t(f"Genel Sonuc: {label}  (Guven: {conf*100:.0f}%)"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)

    pdf.ln(3)
    pdf.set_font(FONT, "B", 9)
    pdf.cell(0, 6, t("Oneri:"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font(FONT, "", 8)
    pdf.multi_cell(0, 5.5, t(rec))

    # ── Medikal uyarı
    pdf.ln(5)
    pdf.set_fill_color(255, 240, 240)
    pdf.set_text_color(160, 0, 0)
    pdf.set_font(FONT, "", 7.5)
    pdf.multi_cell(0, 5, t(MEDICAL_DISCLAIMER), fill=True)
    pdf.set_text_color(0, 0, 0)

    tmp = tempfile.NamedTemporaryFile(
        suffix=".pdf", delete=False, prefix="brakisefali_rapor_"
    )
    pdf.output(tmp.name)
    return tmp.name


# ============================================================
# HTML inşa eden iç fonksiyon
# ============================================================

def _build_html(measurements, classification, patient_info) -> str:
    now = datetime.now().strftime("%d.%m.%Y %H:%M")
    ci    = measurements.get("cephalic_index", 0)
    cvai  = measurements.get("cvai", 0)
    sym   = measurements.get("symmetry_score", 0)
    circ  = measurements.get("circularity", 0)
    overall = classification.get("overall", "?")
    label   = classification.get("label_tr", overall)
    conf    = classification.get("confidence", 0)
    rec     = classification.get("recommendation", "")

    severity_hex = {
        "normal":       "#1a7a3c",
        "mild":         "#c8a000",
        "moderate":     "#c85000",
        "severe":       "#b00000",
        "dolikosefali": "#8a5200",
    }
    color = severity_hex.get(overall, "#555")

    patient_rows = ""
    if patient_info:
        for k, v in patient_info.items():
            if v:
                patient_rows += f"<tr><td><b>{k}</b></td><td>{v}</td></tr>"

    return f"""<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="UTF-8">
  <title>{APP_TITLE} - Rapor</title>
  <style>
    body {{font-family:Arial,sans-serif;margin:30px;color:#222;}}
    h1 {{background:#2980b9;color:#fff;padding:14px 20px;border-radius:6px;}}
    h2 {{color:#2980b9;border-bottom:2px solid #2980b9;padding-bottom:4px;}}
    table {{border-collapse:collapse;width:100%;margin-bottom:18px;}}
    td,th {{border:1px solid #ccc;padding:7px 10px;text-align:left;}}
    th {{background:#eef4ff;}}
    .result {{font-size:1.3em;font-weight:bold;color:{color};
              border:2px solid {color};padding:10px;border-radius:5px;}}
    .warning {{background:#fff0f0;border:1px solid #c00;padding:10px;
               border-radius:5px;color:#900;font-size:.85em;}}
    .rec {{background:#f0fff0;border:1px solid #080;padding:10px;border-radius:5px;}}
    footer {{color:#888;font-size:.8em;margin-top:30px;}}
  </style>
</head>
<body>
  <h1>{APP_TITLE}</h1>
  <p><strong>Analiz Tarihi:</strong> {now} &nbsp;|&nbsp; <strong>Versiyon:</strong> {APP_VERSION}</p>

  {"<h2>Hasta Bilgileri</h2><table>" + patient_rows + "</table>" if patient_rows else ""}

  <h2>Ölçüm Sonuçları</h2>
  <table>
    <tr><th>Ölçüm</th><th>Değer</th><th>Yorum</th></tr>
    <tr><td>Sefalik İndeks (SI)</td><td><b>{ci:.2f}%</b></td>
        <td>{get_ci_interpretation(ci)}</td></tr>
    <tr><td>Krank. Kasa Asimetri İndeksi (KKAI)</td><td><b>{cvai:.2f}%</b></td>
        <td>{get_cvai_interpretation(cvai)}</td></tr>
    <tr><td>Simetri Skoru</td><td>{sym*100:.1f}%</td><td>—</td></tr>
    <tr><td>Dairesellik</td><td>{circ:.3f}</td><td>—</td></tr>
  </table>

  <h2>Sınıflandırma</h2>
  <div class="result">Sonuç: {label} &nbsp;|&nbsp; Güven: {conf*100:.0f}%</div>
  <br>
  <div class="rec"><b>Öneri:</b><br>{rec}</div>

  <h2>Yasal Uyarı</h2>
  <div class="warning">{MEDICAL_DISCLAIMER}</div>

  <footer><p>Bu rapor {APP_TITLE} v{APP_VERSION} tarafından {now} tarihinde oluşturulmuştur.</p></footer>
</body>
</html>"""
