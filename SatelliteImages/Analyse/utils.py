import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.decomposition import PCA
import seaborn as sns
from scipy.stats import norm
from scipy import linalg
import matplotlib as mpl
import pandas as pd


def add_decision_boundary(model, levels=None, resolution=1000, ax=None, label=None, color=None):
    """Trace une frontière de décision sur une figure existante.
                    
    La fonction utilise `model` pour prédire un score ou une classe
    sur une grille de taille `resolution`x`resolution`. Une (ou
    plusieurs frontières) sont ensuite tracées d'après le paramètre
    `levels` qui fixe la valeur des lignes de niveaux recherchées.
    """
    if ax is None:
        ax = plt.gca()
    if callable(model):
        if levels is None:
            levels = [0]
        def predict(X):
            return model(X)
    else:
        n_classes = len(model.classes_)
        if n_classes == 2:
            if hasattr(model, "decision_function"):
                if levels is None:
                    levels = [0]
                def predict(X):
                    return model.decision_function(X)
            elif hasattr(model, "predict_proba"):
                if levels is None:
                    levels = [.5]
                def predict(X):
                    pred = model.predict_proba(X)
                    if pred.shape[1] > 1:
                        return pred[:, 0]
                    else:
                        return pred
            elif hasattr(model, "predict"):
                if levels is None:
                    levels = [.5]
                def predict(X):
                    return model.predict(X)
            else:
                raise Exception("Modèle pas reconnu")
        else:
            levels = np.arange(n_classes - 1) + .5
            def predict(X):
                pred = model.predict(X)
                _, idxs = np.unique(pred, return_inverse=True)
                return idxs
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # Create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], resolution)
    yy = np.linspace(ylim[0], ylim[1], resolution)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = predict(xy).reshape(XX.shape)
    color = "red" if color is None else color
    sns.lineplot([0], [0], label=label, ax=ax, color=color, linestyle="dashed")
    ax.contour(
        XX,
        YY,
        Z,
        levels=levels,
        colors=[color],
        linestyles="dashed",
        antialiased=True,
    )


def plot_clustering(data, labels, markers=None, ax=None, **kwargs):
    """Affiche dans leur premier plan principal les données `data`,
    colorée par `labels` avec éventuellement des symboles `markers`.
    """

    if ax is None:
        ax = plt.gca()

    # Reduce to two dimensions
    if data.shape[1] == 2:
        data_pca = data.to_numpy()
    else:
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data)

    COLORS = np.array(['blue', 'green', 'red', 'purple', 'gray', 'cyan'])
    _, labels = np.unique(labels, return_inverse=True)
    colors = COLORS[labels]

    if markers is None:
        ax.scatter(*data_pca.T, c=colors)
    else:
        MARKERS = "o^sP*+xD"

        # Use integers
        markers_uniq, markers = np.unique(markers, return_inverse=True)

        for marker in range(len(markers_uniq)):
            data_pca_marker = data_pca[markers == marker, :]
            colors_marker = colors[markers == marker]
            ax.scatter(*data_pca_marker.T, c=colors_marker, marker=MARKERS[marker])

    if 'centers' in kwargs and 'covars' in kwargs:
        if data.shape[1] == 2:
            centers_2D = kwargs['centers']
            covars_2D = kwargs['covars']
        else:
            centers_2D = pca.transform(kwargs["centers"])
            covars_2D = [
                pca.components_ @ c @ pca.components_.T
                for c in kwargs['covars']
            ]

        p = 0.9
        sig = norm.ppf(p**(1/2))

        for i, (covar_2D, center_2D) in enumerate(zip(covars_2D, centers_2D)):
            v, w = linalg.eigh(covar_2D)
            print(v)
            v = 2. * sig * np.sqrt(v)

            u = w[0] / linalg.norm(w[0])
            if u[0] == 0:
                angle = np.pi / 2
            else:
                angle = np.arctan(u[1] / u[0])

            color = COLORS[i]
            angle = 180. * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(center_2D, v[0], v[1], 180. + angle, color=color)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.5)
            ax.add_artist(ell)

    return ax

def scatterplot_pca(
    columns=None, hue=None, style=None, data=None, pc1=1, pc2=2, **kwargs
):
    """
    Utilise `sns.scatterplot` en appliquant d'abord une ACP si besoin
    pour réduire la dimension.
    """

    # Select columns (should be numeric)
    data_quant = data if columns is None else data[columns]
    data_quant = data_quant.drop(
        columns=[e for e in [hue, style] if e is not None], errors="ignore"
    )

    # Reduce to two dimensions
    if data_quant.shape[1] == 2:
        data_pca = data_quant
        pca = None
    else:
        n_components = max(pc1, pc2)
        pca = PCA(n_components=n_components)
        data_pca = pca.fit_transform(data_quant)
        data_pca = pd.DataFrame(
            data_pca[:, [pc1 - 1, pc2 - 1]], columns=[f"PC{pc1}", f"PC{pc2}"]
        )

    # Keep name, force categorical data for hue and steal index to
    # avoid unwanted alignment
    if isinstance(hue, pd.Series):
        if not hue.name:
            hue.name = "hue"
        hue_name = hue.name
    elif isinstance(hue, str):
        hue_name = hue
        hue = data[hue]
    elif isinstance(hue, np.ndarray):
        hue = pd.Series(hue, name="class")
        hue_name = "class"

    hue = hue.astype("category")
    hue.index = data_pca.index
    hue.name = hue_name

    if isinstance(style, pd.Series):
        if not style.name:
            style.name = "style"
        style_name = style.name
    elif isinstance(style, str):
        style_name = style
        style = data[style]
    elif isinstance(style, np.ndarray):
        style = pd.Series(style, name="style")
        style_name = "style"

    sp_kwargs = {}
    full_data = data_pca
    if hue is not None:
        full_data = pd.concat((full_data, hue), axis=1)
        sp_kwargs["hue"] = hue_name
    if style is not None:
        full_data = pd.concat((full_data, style), axis=1)
        sp_kwargs["style"] = style_name

    x, y = data_pca.columns
    ax = sns.scatterplot(x=x, y=y, data=full_data, **sp_kwargs)

    return ax, pca

