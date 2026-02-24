# test_box.py
from core.annotation import BoundingBox

print("Testing BoundingBox...")
box = BoundingBox(image_size=(100, 100))
box.from_pixels(10, 10, 50, 50, 100, 100)
print(f"✅ Box created with ID: {box.id}")

# Test copy method
try:
    box2 = box.copy()
    print(f"✅ Copy method works! New box ID: {box2.id}")
except Exception as e:
    print(f"❌ Copy method failed: {e}")

# Test resize method
try:
    handles = box.get_resize_handles()
    print(f"✅ Get resize handles works: {handles.keys()}")
except Exception as e:
    print(f"❌ Get resize handles failed: {e}")

print("\n🎉 Test complete!")