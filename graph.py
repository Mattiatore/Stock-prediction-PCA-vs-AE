import matplotlib.pyplot as plt
import matplotlib
import numpy as np

file = open('values.txt','r') 
text = file.read()
file.close()
righe = text.split("\n")
encod_dim = []
interm_dim = []
pca = []
fae = []
dae = []
righe = righe[:-1]
for riga in righe:
    rig = riga.split()
    encod_dim.append(rig[0])
    interm_dim.append(rig[1])
    pca.append(rig[2])
    fae.append(rig[3])
    dae.append(rig[4])

nrows, ncols = 8,8
image = np.zeros(nrows*ncols)

for i in range(len(pca)):
    minimo = min(pca[i] ,fae[i] ,dae[i])
    if (minimo == fae[i]):
        image[i]=6
    elif (minimo == dae[i]):
        image[i]=10

image = image.reshape((nrows, ncols))

row_labels = range(1,nrows+1)
col_labels = range(2,nrows+2)
plt.matshow(image)
plt.xticks(range(ncols), col_labels)
plt.yticks(range(nrows), row_labels)
font = {'family' : 'DejaVu Sans',
        'weight' : 'bold',
        'size'   : 35}

matplotlib.rc('font', **font)
plt.xlabel('Intermediate AE dimension', fontsize=35)
plt.ylabel('Bottleneck dimension',fontsize=35)
plt.title("MSE comparison between PCA and AE")
plt.rcParams["figure.figsize"] = [20,15]
plt.show()

# didn't find a better way to get a legend
x = np.linspace(0, 20, 1000)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, "purple", label="PCA")
plt.plot(x, y2, "-g", label="FAE")
plt.plot(x, y2, "yellow", label="DAE")
plt.legend(loc="upper left")
plt.ylim(-1.5, 2.0)
plt.show()