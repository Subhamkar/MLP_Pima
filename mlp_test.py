import numpy as np
import mlp


pima = np.loadtxt('C:\Users\subha\PycharmProjects\MLP_Pima/pima-indians-diabetes.data',delimiter=',')

# Plot the first and second values for the two classes
indices0 = np.where(pima[:,8]==0)
indices1 = np.where(pima[:,8]==1)


pima[:,:8] = pima[:,:8]-pima[:,:8].mean(axis=0)
imax = np.concatenate((pima.max(axis=0)*np.ones((1,9)),np.abs(pima.min(axis=0)*np.ones((1,9)))),axis=0).max(axis=0)
pima[:,:8] = pima[:,:8]/imax[:8]
print"\n"


p = mlp.mlp(pima[:,:8],pima[:,8:9],5,outtype='logistic')
p.mlptrain(pima[:,:8],pima[:,8:9],0.25,500)
print"\n"
p.confmat(pima[:,:8],pima[:,8:9])


print "output after preprocessing the data"
train = pima[::2,:8]
traint = pima[::2,8:9]
valid = pima[1::8,0:8]
validt = pima[1::8,8:9]
test = pima[1::2,:8]
testt = pima[1::2,8::9]


import mlp
net = mlp.mlp(train,traint,5,outtype='logistic')
net.earlystopping(train,traint,valid,validt,0.1)
print("\n")
net.confmat(test,testt)
