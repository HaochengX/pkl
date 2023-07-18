import pickle
import time
f = open('neural_obj.pkl','rb')
model = pickle.load(f)
f.close()
f = open('pre_UNSW_NB15_tor.pkl','rb')
data = pickle.load(f)
X, y = data
start_time = time.time()
print('start')
print(model.test(X[1], y[1]))
test_time = time.time() - start_time
print(test_time)
print('end')