from sklearn.decomposition import PCA
from sklearn.utils.extmath import randomized_svd


def do_sign_covariant_pca(X, n_components):
    sums = np.sum(X, axis=1)
    signs = ((sums <= 0) - 0.5) * 2.0
    X_normalized = signs[:, np.newaxis] * X
    U, S, V = randomized_svd(X_normalized,
                             n_components=n_components,
                             flip_sign=True)
    return U * signs[:, np.newaxis]


def do_pca_step(features, n_components, normalize=True, epsilon=1e-8):
    shape_initial = features.shape
    features = np.transpose(features, axes=(0, 2, 3, 1))
    features = np.reshape(features, [-1, features.shape[-1]])
    #features = np.vstack([np.real(features), np.imag(features)])

    if (normalize):
        stds = np.sqrt(np.mean(features * features, axis=0))
        stds = np.maximum(stds, epsilon)
        features = features / stds[np.newaxis, :]

    features = do_sign_covariant_pca(features, n_components)
    #features = features[0:(features.shape[0] // 2)] + 1j * features[(features.shape[0] // 2):]
    features = np.reshape(features, [
        shape_initial[0], shape_initial[2], shape_initial[3],
        features.shape[-1]
    ])
    features = np.transpose(features, axes=(0, 3, 1, 2))
    return features
