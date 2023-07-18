import preprocessor as pre
import pickle
X, y, n_features, n_classes = pre.UNSW_NB15_preprocess()
X_torch, y_torch = pre.to_torch(X, y)
pickle.dump(pre.to_torch(X, y), open("pre_UNSW_NB15_tor.pkl", "wb"))