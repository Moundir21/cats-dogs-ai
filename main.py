import os
import numpy as np
import cv2
from tqdm import tqdm

from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# ⚙️ إعدادات
# -----------------------------
IMG_SIZE = (128, 128)

# -----------------------------
# 🧠 Feature Extraction (HOG + Color)
# -----------------------------
def extract_features(image):
    img = cv2.resize(image, IMG_SIZE)

    # تحسين الصورة
    img = cv2.GaussianBlur(img, (3,3), 0)

    # --- HOG ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hog_feat = hog(
        gray,
        orientations=12,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )

    # --- Color Histogram ---
    hist = cv2.calcHist([img], [0,1,2], None, [8,8,8],
                        [0,256,0,256,0,256])
    hist = cv2.normalize(hist, hist).flatten()

    # --- دمج ---
    return np.hstack([hog_feat, hist])


# -----------------------------
# 📥 تحميل البيانات
# -----------------------------
def load_data(path):
    X, y = [], []

    for label in ["cats", "dogs"]:
        folder = os.path.join(path, label)

        for file in tqdm(os.listdir(folder), desc=f"Loading {label}"):
            try:
                img_path = os.path.join(folder, file)
                img = cv2.imread(img_path)

                if img is None:
                    continue

                features = extract_features(img)
                X.append(features)
                y.append(label)

            except:
                continue

    return np.array(X), np.array(y)


# -----------------------------
# 🚀 MAIN
# -----------------------------
def main():

    print("📥 Loading data...")
    X_train, y_train = load_data("dataset/training_set")
    X_test, y_test = load_data("dataset/test_set")

    print(f"\nShape before PCA: {X_train.shape}")

    # -----------------------------
    # ⚖️ Scaling
    # -----------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -----------------------------
    # 🧠 PCA (تقليل الأبعاد)
    # -----------------------------
    print("\n🔻 Applying PCA...")
    pca = PCA(n_components=200)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    print(f"Shape after PCA: {X_train.shape}")

    # -----------------------------
    # 🤖 تجربة عدة قيم لـ K
    # -----------------------------
    print("\n🔍 Searching best K...")

    best_acc = 0
    best_k = 1

    for k in [1, 3, 5, 7]:
        knn = KNeighborsClassifier(
            n_neighbors=k,
            weights='distance',
            metric='euclidean'
        )

        knn.fit(X_train, y_train)
        pred = knn.predict(X_test)
        acc = accuracy_score(y_test, pred)

        print(f"k={k} → Accuracy={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_k = k

    # -----------------------------
    # 🏆 أفضل موديل
    # -----------------------------
    print("\n🏆 Best Model:")
    print(f"k = {best_k} | Accuracy = {best_acc:.4f}")

    knn = KNeighborsClassifier(
        n_neighbors=best_k,
        weights='distance',
        metric='euclidean'
    )

    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    # -----------------------------
    # 📊 النتائج
    # -----------------------------
    print("\n📊 Final Results")
    print("-" * 40)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print("-" * 40)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # -----------------------------
    # 💾 حفظ النتائج
    # -----------------------------
    os.makedirs("results", exist_ok=True)

    with open("results/final_results.txt", "w") as f:
        f.write(f"Best k: {best_k}\n")
        f.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")


if __name__ == "__main__":
    main()