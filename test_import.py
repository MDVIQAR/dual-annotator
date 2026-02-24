# test_import.py
print("Testing imports...")

try:
    from core.class_manager import ClassManager, ClassCategory
    print("✅ Successfully imported ClassManager and ClassCategory")
except Exception as e:
    print(f"❌ Import failed: {e}")

try:
    from core.annotation import BoundingBox
    print("✅ Successfully imported BoundingBox")
except Exception as e:
    print(f"❌ Import failed: {e}")

try:
    from gui.class_panel import ClassPanel
    print("✅ Successfully imported ClassPanel")
except Exception as e:
    print(f"❌ Import failed: {e}")

print("\nTest complete!")