import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from numba import jit
from tools.BasePF import *
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat, savemat

class HarmonicPF(BasePF):
    def __init__(self,fname = None,obs_space_size=None,target_pos=None,obstacles=None,h=0.5):
        if fname != None:            
            mat = loadmat(fname, squeeze_me = True)
            
            if mat['name'] != 'HarmonicPF':
                raise ValueError('The name should be "HarmonicPF"')
            self.name = mat['name']
            self.x = mat['x']
            self.y = mat['y']
            self.xx = mat['xx']
            self.yy = mat['yy']
            self.zz = mat['zz']
            self.target = mat['target']
            self.target_idx = mat['target_idx']
            self.obst_idx = mat['obst_idx']
            self.h = mat["h"]

            super().__init__(
                obs_space_size = np.array([self.x[0,-1], self.y[0,-1]]),
                target=self.target,
                obstacles=obstacles,
                h=self.h
                )

            self.zinterpfun = RegularGridInterpolator((self.x, self.y), self.zz, bounds_error=False)

            dzdx = np.gradient(self.zz,self.h,axis = 1)
            dzdy = np.gradient(self.zz,self.h,axis = 0)
            self.dzinterpfun = [
                RegularGridInterpolator((self.x, self.y), dzdx, bounds_error=False),
                RegularGridInterpolator((self.x, self.y), dzdy, bounds_error=False)
            ]
        else:

            super().__init__(obs_space_size,target_pos,obstacles)
            
            self.name = 'HarmonicPF' # Overriding the name.

            self.h = h
            self.x = np.arange(obs_space_size[0],obs_space_size[1],h)
            self.y = np.arange(obs_space_size[0],obs_space_size[1],h)
            self.xx, self.yy = np.meshgrid(self.x,self.y,indexing='xy')
            self.zz = np.zeros_like(self.xx,dtype=np.float64)
            
            # Convert the target position to the index of the grid.
            self.target_idx = np.unravel_index(
                np.argmin(
                    (self.xx-self.target_pos[0])**2 + (self.yy - self.target_pos[1])**2), 
                self.xx.shape)
            
            # Convert the obstacles' position to the index of the grid.
            self.obst_idx = obstacles.is_collision(np.array([self.xx.flatten(),self.yy.flatten()]))
            self.obst_idx = self.obst_idx.reshape(self.xx.shape[0],self.xx.shape[1])

            # Add the boundary as the obstacles.
            self.obst_idx[0,:] = True
            self.obst_idx[:,0] = True
            self.obst_idx[-1,:] = True
            self.obst_idx[:,-1] = True

        obst_uplo_indx = np.logical_xor(self.obst_idx[1:,:],self.obst_idx[:-1,:])
        self.obst_lo_indx = np.logical_and(obst_uplo_indx,self.obst_idx[1:,:])
        self.obst_up_indx = np.logical_and(obst_uplo_indx,self.obst_idx[:-1,:])

        obst_lr_indx = np.logical_xor(self.obst_idx[:,1:],self.obst_idx[:,:-1])
        self.obst_l_indx = np.logical_and(obst_lr_indx,self.obst_idx[:,1:])
        self.obst_r_indx = np.logical_and(obst_lr_indx,self.obst_idx[:,:-1])
        
    @jit
    def calculate_field(self):
        k = 0
        l_zz = np.ones_like(self.zz)
        while True:
            # Finite element method for laplace equation.
            zz_prev = np.copy(l_zz)
            for i in range(1,l_zz.shape[0]-1):
                for j in range(1,l_zz.shape[1]-1):
                        l_zz[i,j] = (l_zz[i,j+1] + l_zz[i,j-1] + l_zz[i+1,j] +l_zz[i-1,j])/4

            # Dirichlet Boundary Condition.
            l_zz[*self.target_idx] = 0

            # Neumann Boundary Condition.
            l_zz[:,1:][self.obst_r_indx] = l_zz[:,:-1][self.obst_r_indx]
            l_zz[:,:-1][self.obst_l_indx] = l_zz[:,1:][self.obst_l_indx]
            l_zz[1:,:][self.obst_up_indx] = l_zz[:-1,:][self.obst_up_indx]
            l_zz[:-1,:][self.obst_lo_indx] = l_zz[1:,:][self.obst_lo_indx]

            perc = 1e-5/np.max(np.abs(zz_prev - l_zz)) * 1e2

            if k % 100 == 0:
                sys.stdout.write('%d%%\r' %(perc))
                sys.stdout.flush()
            if perc > 1e2:
                break
            
            k += 1
            
        self.zz = l_zz
        # self.zinterpfun = RegularGridInterpolator((self.x, self.y), self.zz, bounds_error=False)

        # dzdx = np.gradient(self.zz,self.h,axis = 1)
        # dzdy = np.gradient(self.zz,self.h,axis = 0)
        # self.dzinterpfun = [
        #     RegularGridInterpolator((self.x, self.y), dzdx, bounds_error=False),
        #     RegularGridInterpolator((self.x, self.y), dzdy, bounds_error=False)
        # ]

    def zfun(self,x):
        return np.squeeze(self.zinterpfun(x))

    def dzfun(self,x):
        return [-self.dzinterpfun[0](x)[0],-self.dzinterpfun[1](x)[0]]

    def plot(self,ax = None):
        if ax is None:
            fig, ax = plt.subplots()

        ctr = ax.contourf(self.xx,self.yy,self.zz, cmap=cm.coolwarm)

        return ctr

    def streamplot(self):
        dzdx = np.gradient(self.zz,self.h,axis = 1)
        dzdy = np.gradient(self.zz,self.h,axis = 0)
        plt.streamplot(self.x,self.y,-dzdx,-dzdy)

    def quiver(self):
        x = np.linspace(self.x[0],self.x[-1],50)
        y = np.linspace(self.y[0],self.y[-1],50)

        dzdx = np.zeros([x.shape[0],y.shape[0]])
        dzdy = np.zeros([x.shape[0],y.shape[0]])
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                dzdx[i,j] = self.dzinterpfun[0]([x[i],y[j]])
                dzdy[i,j] = self.dzinterpfun[1]([x[i],y[j]])

        plt.quiver(x,y,-dzdx,-dzdy)

    def save(self,fname):
        mat = {
            'x': self.x,
            'y': self.y,
            'xx': self.xx,
            'yy': self.yy,
            'zz': self.zz,
            'target': self.target,
            'target_idx': self.target_idx,
            'obst_idx': self.obst_idx,
            'h': self.h
        }
        savemat(fname,mat)

if __name__ == "__main__":
    obsts_list = [
        {'xyt': [20, 10, 0], 'size': [5, 5]},
        {'xyt': [15, 35, 0], 'size': [1,10]},
        {'xyt': [20, 40, 0], 'size': [30,1]},
        {'xyt': [45, 30, 0], 'size': [10,1]},
        {'xyt': [25, 20, 0], 'size': [20,1]},
        {'xyt': [30, 10, 0], 'size': [1,20]}
        ]
    obsts = Obstacles(obsts_list)
    
    hpf = HarmonicPF(
        obs_space_size = np.array([0,50,0,50]), 
        target_pos = np.array([45,45,0]),
        obstacles=obsts,
        h = 0.25)
    hpf.calculate_field()
    hpf.save("vecfield.mat")

    # hpf = HarmonicPF(fname = "vecfield.mat")
    # hpf.plot()
    # hpf.streamplot()
    # ax = plt.gca()
    # obsts.plot(ax)
    # plt.show()