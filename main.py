"""
main.py - Uygulama giriş noktası

Kullanım:
  python main.py                  # Gradio web arayüzü (varsayılan)
  python main.py --cli img.jpg    # Komut satırı analizi
  python main.py --test           # Kısa kütüphane test kontrolü
"""

import sys
import os
import argparse
import logging
import gradio as gr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_web():
    """Gradio web arayüzünü başlatır."""
    from app import build_interface
    demo = build_interface()
    logger.info("Gradio arayüzü başlatılıyor: http://localhost:7860")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    )


def run_cli(image_path: str, view: str = "top"):
    """Komut satırından tek görüntü analizi."""
    from src.preprocessor import load_image_from_path
    from src.analyzer import analyze

    if not os.path.isfile(image_path):
        print(f"[HATA] Dosya bulunamadı: {image_path}")
        sys.exit(1)

    print(f"Analiz ediliyor: {image_path}  (görünüm={view})")
    img = load_image_from_path(image_path)
    result = analyze(img, view=view)

    print("\n" + result.get("text_report", "Rapor oluşturulamadı."))

    if result.get("pdf_path"):
        print(f"\nPDF rapor: {result['pdf_path']}")

    if result.get("annotated_img") is not None:
        out_path = image_path.rsplit(".", 1)[0] + "_analiz.jpg"
        import cv2
        ann = result["annotated_img"]
        cv2.imwrite(out_path, cv2.cvtColor(ann, cv2.COLOR_RGB2BGR))
        print(f"Anotasyonlu görüntü: {out_path}")


def run_test():
    """Temel kütüphane varlık testleri."""
    import importlib
    tests = [
        ("cv2",              "OpenCV"),
        ("numpy",            "NumPy"),
        ("PIL",              "Pillow"),
        ("gradio",           "Gradio"),
        ("scipy",            "SciPy"),
        ("sklearn",          "scikit-learn"),
        ("matplotlib",       "Matplotlib"),
        ("mediapipe",        "MediaPipe"),
        ("rembg",            "rembg (opsiyonel)"),
        ("fpdf",             "fpdf2 (opsiyonel)"),
        ("reportlab",        "ReportLab (opsiyonel)"),
    ]
    print("=" * 50)
    print("  Kütüphane Kontrol Testi")
    print("=" * 50)
    all_ok = True
    for mod, name in tests:
        try:
            importlib.import_module(mod)
            print(f"  [OK]       {name}")
        except ImportError:
            marker = "[OPSIYONEL]" if "opsiyonel" in name else "[EKSİK]   "
            colour_ok = "opsiyonel" in name
            print(f"  {marker} {name}")
            if not colour_ok:
                all_ok = False

    print("=" * 50)
    if all_ok:
        print("  Tüm zorunlu kütüphaneler mevcut. ✓")
    else:
        print("  Bazı zorunlu kütüphaneler eksik!")
        print("  Lütfen: pip install -r requirements.txt")
    print()


# ============================================================
# Argüman ayrıştırma
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deformasyonel Brakisefali Tespit Sistemi",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--cli",
        metavar="IMAGE",
        help="Komut satırı analizi için görüntü dosyası yolu",
    )
    parser.add_argument(
        "--view",
        choices=["top", "front", "side"],
        default="top",
        help="Görünüm türü: top (varsayılan) | front | side",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Kütüphane test kontrolü yap ve çık",
    )

    args = parser.parse_args()

    if args.test:
        run_test()
    elif args.cli:
        run_cli(args.cli, view=args.view)
    else:
        run_web()
