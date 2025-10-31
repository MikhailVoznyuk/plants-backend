# tools/gen_tests.py
import os, math
from PIL import Image, ImageDraw, ImageFilter
import random

def save(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path, "JPEG", quality=92)

def green_bg(w, h):
    return Image.new("RGB", (w, h), (20, 150, 60))

def add_brown_patch(img, box, intensity=1.0):
    d = ImageDraw.Draw(img)
    # brown-ish rectangle with soft edges
    x1,y1,x2,y2 = box
    d.rectangle([x1,y1,x2,y2], fill=(120, 70, int(40*intensity)))
    return img

def add_noise(img, amp=8):
    import random
    px = img.load()
    w,h = img.size
    for y in range(h):
        for x in range(w):
            r,g,b = px[x,y]
            dr = random.randint(-amp, amp)
            dg = random.randint(-amp, amp)
            db = random.randint(-amp, amp)
            px[x,y] = (min(255,max(0,r+dr)),min(255,max(0,g+dg)),min(255,max(0,b+db)))
    return img

def tilted_tree_hint(img, center, size, angle_deg, color=(0,120,0)):
    # draw a simple "plant-like" blob to help fallbacks
    d = ImageDraw.Draw(img)
    cx,cy = center
    w,h = size
    # draw ellipse, then rotate whole image small overlay
    leaf = Image.new("RGBA", (w*2,h*2), (0,0,0,0))
    dl = ImageDraw.Draw(leaf)
    dl.ellipse([w//2, h//2, w + w//2, h + h//2], fill=color+(255,))
    leaf = leaf.rotate(angle_deg, expand=True, resample=Image.BICUBIC)
    img.paste(leaf, (cx-w, cy-h), leaf)
    return img

def gen_all(dst="tests"):
    os.makedirs(dst, exist_ok=True)

    # 1. Plain green 640x480 (should produce PLANT_FALLBACK plants if enabled)
    img = green_bg(640, 480)
    save(img, os.path.join(dst, "t1_green_640.jpg"))

    # 2. Green with brown patch inside (should produce a defect via DEFECT_FALLBACK when plants exist)
    img = green_bg(640, 480)
    img = add_brown_patch(img, (220,160,420,320), intensity=1.0)
    save(img, os.path.join(dst, "t2_green_brown.jpg"))

    # 3. High-res 4000x3000 to catch resize/broadcast bugs
    img = green_bg(4000, 3000)
    img = add_brown_patch(img, (1800,1200,2200,1600), intensity=1.0)
    save(img, os.path.join(dst, "t3_big_4000x3000.jpg"))

    # 4. Tilted hint blob (to see tilt sign)
    img = green_bg(800, 600)
    img = tilted_tree_hint(img, (400,300), (120,220), angle_deg=25)
    save(img, os.path.join(dst, "t4_tilt_plus25.jpg"))

    # 5. Tilted other direction
    img = green_bg(800, 600)
    img = tilted_tree_hint(img, (400,300), (120,220), angle_deg=-30)
    save(img, os.path.join(dst, "t5_tilt_minus30.jpg"))

    # 6. Low-noise scene
    img = add_noise(green_bg(800,600))
    save(img, os.path.join(dst, "t6_noise.jpg"))

    print("Generated test images in:", dst)

if __name__ == "__main__":
    gen_all()
