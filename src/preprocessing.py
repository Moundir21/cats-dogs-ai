import cv2

def preprocess_image(img, size=(128, 128)):

    # Resize
    img = cv2.resize(img, size)

    # Gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE (تحسين الإضاءة 🔥)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    return gray, img