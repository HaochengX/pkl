import pickle
import time
f = open('mlp_model.pkl','rb')
model = pickle.load(f)
f.close()
f = open('pre_UNSW_NB15.pkl','rb')
data = pickle.load(f)
X, y, n_features, n_classes = data
start_time = time.time()
print('start')
print(model.score(X[1], y[1]))
test_time = time.time() - start_time
print(test_time)
print('end')