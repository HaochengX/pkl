import pickle
import time
from sklearn.metrics import accuracy_score
f = open('svc_model.pkl','rb')
model = pickle.load(f)
f.close()
f = open('pre_UNSW_NB15.pkl','rb')
data = pickle.load(f)
X, y, n_features, n_classes = data
start_time = time.time()
print('start')
#model.predict(X[1])
print(accuracy_score(y[1], model.predict(X[1])))
test_time = time.time() - start_time
print(test_time)
print('end')