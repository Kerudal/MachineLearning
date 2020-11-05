from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.utils import check_X_y


def accuracy(X_train, y_train, X_val, y_val, n_neighbors):
    """Précision d'un modèle Knn pour un jeu de données
    d'apprentissage et de validation fournis."""

    # Définition, apprentissage et prédiction par la méthode des
    # plus proches voisins avec `n_neighbors` voisins
    cls = KNeighborsClassifier(n_neighbors=n_neighbors)
    cls.fit(X_train, y_train)
    pred = cls.predict(X_val)

    # Calcul de la précision avec `accuracy_score`
    acc = accuracy_score(pred, y_val)

    return acc


def knn_simple_validation(X_train, y_train, X_val, y_val, n_neighbors_list):
    """Génère les couples nombres de voisins et précision
    correspondante sur l'ensemble de validation."""

    # Calcul des précisions pour tous les nombres de voisins présents
    # dans `n_neighbors_list`
    n = X_train.shape[0]
    for n_neighbors in n_neighbors_list:
        yield (
            n_neighbors,
            accuracy(X_train, y_train, X_val, y_val, n_neighbors),
            n / n_neighbors
        )


def knn_multiple_validation(X, y, n_splits, train_size, n_neighbors_list):
    """Génère les couples nombre de voisins et précisions correspondantes."""

    # Conversion en tableau numpy si on fournit des DataFrame par exemple
    X, y = check_X_y(X, y)

    def models_accuracies(train_index, val_index, n_neighbors_list):
        """Précision de tous les modèles pour un jeu de données fixé."""

        # Création de `X_train`, `y_train`, `X_val` et `y_val`
        X_train = X[train_index, :]
        y_train = y[train_index]
        X_val = X[val_index, :]
        y_val = y[val_index]

        # Calcul des précisions pour chaque nombre de voisins présent
        # dans `n_neighbors`
        n = len(train_index)
        for n_neighbors in n_neighbors_list:
            yield (
                n_neighbors,
                accuracy(X_train, y_train, X_val, y_val, n_neighbors),
                n / n_neighbors
            )

    # Définition de `n_splits` jeu de données avec `ShuffleSplit`
    ms = ShuffleSplit(n_splits=n_splits, train_size=train_size).split(X)

    # Calcul et retour des précisions avec `models_accuracies` pour
    # chaque jeu de données défini par `ShuffleSplit`.
    for train_index, test_index in ms:
        yield from models_accuracies(train_index, test_index, n_neighbors_list)
