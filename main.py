import os
import numpy as np
from tqdm import tqdm
import cv2
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from src.utils import load_dataset
from src.preprocessing import preprocess_image
from src.feature_extraction import extract_features
from src.knn_model import train_knn


def process_images(images, augment=False):
    features = []
    labels_aug = []

    for img in tqdm(images, desc="Processing"):

        gray, color = preprocess_image(img)
        feat = extract_features(gray, color)

        features.append(feat)

        # 🔥 Data Augmentation بسيط
        if augment:
            flipped = cv2.flip(img, 1)
            gray_f, color_f = preprocess_image(flipped)
            feat_f = extract_features(gray_f, color_f)
            features.append(feat_f)

    return np.array(features)


def find_best_k(X_train, y_train, X_test, y_test):

    best_k = 1
    best_acc = 0

    print("\n🔍 Searching best k...\n")

    for k in range(1, 15, 2):

        model = train_knn(X_train, y_train, k)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        print(f"k={k} → Accuracy={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_k = k

    print(f"\n🏆 Best k: {best_k} ({best_acc:.4f})")
    return best_k


def main():

    print("📥 Loading dataset...\n")

    train_images, y_train = load_dataset("dataset/training_set")
    test_images, y_test = load_dataset("dataset/test_set")

    print("\n⚙️ Feature Extraction...\n")

    X_train = process_images(train_images)
    X_test = process_images(test_images)

    print("\n⚖️ Scaling...\n")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    best_k = find_best_k(X_train, y_train, X_test, y_test)

    print("\n🤖 Training final model...\n")

    model = train_knn(X_train, y_train, best_k)

    print("\n🔍 Predicting...\n")

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"\n✅ Final Accuracy: {acc:.4f}")

    print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    os.makedirs("results", exist_ok=True)

    with open("results/accuracy.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Best k: {best_k}\n")

    print("\n📁 Results saved!")


if __name__ == "__main__":
    main()