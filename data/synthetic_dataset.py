import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
class SyntheticData(object):
    """ Data structure for syntetic 2 dimensional data
        
    """
    def __init__(self, cid, data=None, update_id=True):
        self.cid = cid
        if data is None:
            self.data = pd.DataFrame(columns=['x','y', 'id'])
        else:
            self.data = data
            if update_id: 
                self.data['id'] = self.cid
        
    def __add__(self, other):
        """
        add two SynteticData sets and make a new including all points of those two sets.
        the cid will be string combined of cids of original sets
        """
        return SyntheticData(cid = self.cid + '+' + other.cid, data = self.data.append(other.data, ignore_index=True), update_id=False)
    
    def __repr__(self):
        return "data of cluster "+ str(self.cid) +"\n" + self.data.to_string()
 
    def update_cid(self, cid=None):
        if cid is None:
            self.data['id'] = self.cid 
        else:
            self.cid = cid
            self.data['id'] = cid
   
    def plot(self, **kwargs):
        plt.scatter(self.data.x, self.data.y, label=self.cid, **kwargs)
        plt.legend()
        
    def generate(self, **kwargs):
        raise NotImplementedError()
        
    def rotate(self, angel):
        rotation = [[np.cos(angel), np.sin(angel)], [-np.sin(angel), np.cos(angel)]]
        data = pd.DataFrame(data= np.matmul(rotation, self.data[['x','y']].to_numpy().transpose()).transpose(), 
                                 columns=['x','y'])
        self.data.x = data.x
        self.data.y = data.y
        
        
    def shift(self, center):
        self.data.x += center[0]
        self.data.y += center[1]
        
    def stretch(self, scales):
        self.data.x *= scales[0]
        self.data.y *= scales[1]
        
    def add_noise(self, sigma):
        rnd = np.random.randn(2, len(self.data.x))
        self.data.x += sigma*rnd[0]
        self.data.y += sigma*rnd[1]
        
        
class UniformRnd(SyntheticData):
    """
    """
    def generate(self, number_points, scales = [1.0, 1.0]):
        self.data = pd.DataFrame(columns=['x','y', 'id'])
        rnd_num = np.random.uniform(size=[2, number_points])
        self.data.x, self.data.y = scales[0]*rnd_num[0], scales[1]*rnd_num[1] 
        self.data.id = self.cid

class RingRnd(SyntheticData):
    def generate(self, number_points, radius=[0.0, 1.0], angel=[0.0, 2.0*np.pi], scales=[1.0,1.0]):
        self.data = pd.DataFrame(columns=['x','y','id'])
        radius_rnd = np.random.uniform(low= radius[0], high=radius[1], size=number_points)
        theta_rnd = np.random.uniform(low= angel[0], high=angel[1],size=number_points)
        self.data.x = scales[0]*radius_rnd*np.cos(theta_rnd)
        self.data.y = scales[1]*radius_rnd*np.sin(theta_rnd)
        self.data.id = self.cid

def plot_clusters(*clusters, **kwargs):
    for cls in clusters:        
        plt.scatter(cls[0].data.x, cls[0].data.y, label=cls[0].cid, c = cls[1], **kwargs)
    plt.legend()
    
def syntheticdataset(i = 'level1_far'):
    np.random.seed(0)
    if i == 'level1_far':
        normal = RingRnd(0)
        normal.generate(number_points=10000,)
        outliers = RingRnd(1)
        outliers.generate(number_points=int(0.005*10000), radius=[1.2, 2.0])
        
    if i == 'level1_close':
        normal = RingRnd(0)
        normal.generate(number_points=10000,)
        normal.add_noise(0.5)
        outliers = RingRnd(1)
        outliers.generate(number_points=int(0.005*10000), radius=[1.2, 2.0])
        outliers.add_noise(0.5)

    if i == 'level1_cluster':
        normal = RingRnd(0)
        normal.generate(number_points=10000,)
        outliers = RingRnd(1)
        outliers.generate(number_points=int(0.005*10000), radius=[0.0, 0.2])
        outliers.shift([1.2, 0])

    if i == 'level1_cluster2':
        normal = RingRnd(0)
        normal.generate(number_points=10000,)
        outliers1 = RingRnd('outlier1')
        outliers1.generate(number_points=int(0.005*10000/2), radius=[0.0, 0.2])
        outliers1.shift([1.2, 0])
        outliers2 = RingRnd('outlier2')
        outliers2.generate(number_points=int(0.005*10000/2), radius=[0.0, 0.2])
        outliers2.shift([-1.2, 0])
        outliers = outliers1 + outliers2
        outliers.update_cid(1)

    if i == 'level2_close':
        normal = RingRnd(0)
        normal.generate(number_points=int(10000), radius=[0.0, 1.0], scales=[2.5,0.2])
        normal.rotate(np.pi/4.0)
        outliers = RingRnd(1)
        outliers.generate(number_points=int(0.005*10000), radius=[1.2, 2.0], scales=[3.5,0.3])
        outliers.rotate(np.pi/4.0)

    if i == 'level3_close':
        normal = RingRnd(0)
        normal.generate(number_points=10000, radius=[0.5, 1.5])
        outliers = RingRnd(1)
        outliers.generate(number_points=int(0.005*10000), radius=[0.0, 0.2])

    if i == 'level3_far':
        normal = RingRnd(0)
        normal.generate(number_points=10000, radius=[0.5, 1.0])
        outliers1 = RingRnd('outlier1')
        outliers1.generate(number_points=int(0.005*10000/2), radius=[0.0, 0.2])
        outliers2 = RingRnd('outlier2')
        outliers2.generate(number_points=int(0.005*10000/2), radius=[1.3, 2.0])
        outliers = outliers1 + outliers2
        outliers.update_cid(1)

    if i == 'level4_far':
        normal1 = RingRnd('normal1')
        normal1.generate(number_points=int(10000/2), radius=[0.0, 1.0])
        normal1.shift([1.5,0.0])

        normal2 = RingRnd('normal2')
        normal2.generate(number_points=int(10000/2), radius=[0.0, 1.0])
        normal2.shift([-1.5,0.0])

        normal3 = RingRnd('normal3')
        normal3.generate(number_points=int(10000/5), radius=[0.0, 1.0], scales=[2.5,0.2])


        normal = normal1 + normal2 + normal3
        normal.update_cid(0)

        outliers1 = RingRnd('outlier1')
        outliers1.generate(number_points=int(0.005*10000/2), radius=[0.0, 0.2])
        outliers1.shift([0.0, 0.5])
        outliers2 = RingRnd('outlier2')
        outliers2.generate(number_points=int(0.005*10000/2), radius=[0.0, 0.2])
        outliers2.shift([0.0, -0.5])
        outliers = outliers1 + outliers2
        outliers.update_cid(1)
        
    if i == 'level4_close':
        normal1 = RingRnd('normal1')
        normal1.generate(number_points=int(10000/2), radius=[0.0, 1.0], angel=[np.pi/6, np.pi*(2-1/6)])
        normal1.shift([1.5,0.0])

        normal2 = RingRnd('normal2')
        normal2.generate(number_points=int(10000/2), radius=[0.0, 1.0], angel=[-np.pi*(1-1/6), np.pi*(1-1/6)])
        normal2.shift([-1.5,0.0])

        normal3 = RingRnd('normal3')
        normal3.generate(number_points=int(10000/5), radius=[0.0, 1.0], scales=[2.5,0.2])


        normal = normal1 + normal2 + normal3
        normal.update_cid(0)
        normal.add_noise(0.1)

        outliers1 = RingRnd('outlier1')
        outliers1.generate(number_points=int(0.005*10000/2), radius=[0.0, 0.2])
        outliers1.shift([0.0, 0.5])
        outliers2 = RingRnd('outlier2')
        outliers2.generate(number_points=int(0.005*10000/2), radius=[0.0, 0.2])
        outliers2.shift([0.0, -0.5])
        outliers = outliers1 + outliers2
        outliers.update_cid(1)
        
    X = normal.data.append(outliers.data)[['x', 'y']]
    y = normal.data.append(outliers.data)['id']

    return X, y