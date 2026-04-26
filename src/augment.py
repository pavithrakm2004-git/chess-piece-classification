import cv2
import os
import numpy as np
from pathlib import Path

PROCESSED_DIR = r"C:\Users\User\OneDrive\Desktop\chess-piece-classifier\data\processed"
TARGET = 200  # images per class we want

def augment_image(img):
    results = []

    # Horizontal flip
    results.append(cv2.flip(img, 1))

    # Brightness up
    results.append(cv2.convertScaleAbs(img, alpha=1.3, beta=20))

    # Brightness down
    results.append(cv2.convertScaleAbs(img, alpha=0.7, beta=-20))

    # Rotate 10 degrees
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), 10, 1.0)
    results.append(cv2.warpAffine(img, M, (w, h)))

    # Rotate -10 degrees
    M2 = cv2.getRotationMatrix2D((w//2, h//2), -10, 1.0)
    results.append(cv2.warpAffine(img, M2, (w, h)))

    # Add noise
    noise = np.random.randint(0, 25, img.shape, dtype=np.uint8)
    results.append(cv2.add(img, noise))

    return results

def augment_dataset():
    processed_path = Path(PROCESSED_DIR)
    classes = [d for d in processed_path.iterdir() if d.is_dir()]

    print("=== AUGMENTATION ===\n")

    for cls_dir in sorted(classes):
        images = list(cls_dir.glob("*.jpg")) + \
                 list(cls_dir.glob("*.jpeg")) + \
                 list(cls_dir.glob("*.png"))

        current = len(images)
        needed  = max(0, TARGET - current)

        print(f"{cls_dir.name:20s}: {current} → ", end="")

        if needed == 0:
            print(f"{current} (no augmentation needed)")
            continue

        aug_count = 0
        i = 0
        while aug_count < needed:
            img_path = images[i % len(images)]
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                i += 1
                continue

            augmented = augment_image(img)
            for aug_img in augmented:
                if aug_count >= needed:
                    break
                out_name = f"aug_{aug_count:04d}_{img_path.name}"
                cv2.imwrite(str(cls_dir / out_name), aug_img)
                aug_count += 1

            i += 1

        final = len(list(cls_dir.glob("*.jpg")) +
                    list(cls_dir.glob("*.jpeg")) +
                    list(cls_dir.glob("*.png")))
        print(f"{final} ✓")

    print("\nDone! All classes now have", TARGET, "images.")

if __name__ == "__main__":
    augment_dataset()