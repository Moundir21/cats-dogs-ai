from sklearn.neighbors import KNeighborsClassifier

def train_knn(X, y, k):

    model = KNeighborsClassifier(
        n_neighbors=k,
        metric='euclidean',
        weights='distance',   # 🔥 مهم
        n_jobs=-1
    )

    model.fit(X, y)
    return model