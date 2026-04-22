import os
import cv2

def load_dataset(base_path):
    images = []
    labels = []

    classes = ["cats", "dogs"]

    for label, cls in enumerate(classes):
        folder = os.path.join(base_path, cls)

        if not os.path.exists(folder):
            print(f"❌ Folder not found: {folder}")
            continue

        for file in os.listdir(folder):
            path = os.path.join(folder, file)

            if not file.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            img = cv2.imread(path)

            if img is None:
                continue

            images.append(img)
            labels.append(label)

    print(f"✅ Loaded {len(images)} images from {base_path}")
    return images, labels