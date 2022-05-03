import scipy.io as scio

dataFile = 'F:\DATA\PointCloud\i/000000.mat'
data = scio.loadmat(dataFile)
print(data)
print(data['depth'])
print (type(data['depth']))