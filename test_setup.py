# test_setup.py
import sys
print(f"Python version: {sys.version}")
print(f"Virtual env: {sys.prefix}")
print("\n--- Testing Libraries ---")

try:
    import PyQt5
    print("✅ PyQt5 installed")
except ImportError as e:
    print(f"❌ PyQt5 failed: {e}")

try:
    import numpy
    print(f"✅ NumPy {numpy.__version__} installed")
except ImportError as e:
    print(f"❌ NumPy failed: {e}")

try:
    import cv2
    print(f"✅ OpenCV {cv2.__version__} installed")
except ImportError as e:
    print(f"❌ OpenCV failed: {e}")

try:
    from PIL import Image
    print(f"✅ Pillow {Image.__version__} installed")
except ImportError as e:
    print(f"❌ Pillow failed: {e}")

try:
    import shapely
    print(f"✅ Shapely {shapely.__version__} installed")
except ImportError as e:
    print(f"❌ Shapely failed: {e}")

try:
    from pycocotools import coco
    print("✅ pycocotools installed")
except ImportError as e:
    print(f"❌ pycocotools failed: {e}")

print("\n🎉 Setup complete! Ready for Day 1!")