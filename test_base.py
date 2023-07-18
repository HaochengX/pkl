import pickle
import time
f = open('baselineHD_model.pkl','rb')
model = pickle.load(f)
f.close()
f = open('pre_UNSW_NB15_tor.pkl','rb')
data = pickle.load(f)
X, y = data
start_time = time.time()
print('start')
y_pred = model(X[1])
print(((y_pred == y[1]).sum() / (y_pred.size(0))).item())
test_time = time.time() - start_time
print(test_time)
print('end')