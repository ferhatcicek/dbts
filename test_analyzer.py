"""Test: Analyzer modülü uçtan uca"""
import sys, numpy as np, cv2
sys.path.insert(0, ".")
from src.analyzer import analyze

img = np.ones((500, 500, 3), dtype=np.uint8) * 200
cv2.ellipse(img, (250, 250), (175, 190), 0, 0, 360, (80, 60, 50), -1)

print("Tam pipeline analizi...", flush=True)
result = analyze(img, view="top")

success = result["success"]
ann = result["annotated_img"]
gauge = result["gauge_img"]
pdf = result["pdf_path"]
report_len = len(result["text_report"])

print(f"Basari         : {success}")
print(f"Annotated sekli: {ann.shape if ann is not None else 'None'}")
print(f"Gauge sekli    : {gauge.shape if gauge is not None else 'None'}")
print(f"PDF yolu       : {pdf}")
print(f"Rapor uzunlugu : {report_len} karakter")
print()
print("=== ANALYZER TEST BASARILI ===" if success else "=== TEST BASARISIZ ===")
