import numpy as np 
import time



class Optimizer:  
    def __init__(self, save_f=False, *args, **kwargs):
        self.save_f = save_f
        if save_f:
            self.list_t = []
            self.list_f = []
    
    def step(self, x):
        pass
    
    def solve(self, x0, tol, max_iter):
        x = x0
        convergence_criterion = False
        n_iter = 0
        self.start = time.time()
        while not convergence_criterion and n_iter < max_iter:
            n_iter += 1
            old_x = x.copy()
            x = self.step(x)
            convergence_criterion = np.linalg.norm(old_x -x) < tol
        end = time.time()
        print(f"Duration : {end - self.start}")
        return x
    
    
    

    
class PlugAndPlayADMM(Optimizer):
    
    def __init__(self, x0, A, rho, proj, net, save_f=False):
        super().__init__(save_f)
        self.n = int(np.sqrt(x0.shape[0]))
        self.A = A
        self.rho = rho
        self.proj = proj
        self.pseudo_inv = np.linalg.inv( (A.T).dot(A) + rho*np.eye(A.shape[1]) )
        self.At_proj = np.squeeze(np.array( (A.T).dot(proj) ))
        self.net = net
        self.x_tilde = x0.copy()
        self.v_tilde = x0.copy()
        self.u_bar = np.ones(x0.shape[0])
        
        
    def step(self, x):
        x = self.pseudo_inv.dot(self.At_proj + self.rho*self.x_tilde)
        x = np.squeeze(np.array(x))
        v = denoising_image(self.v_tilde.reshape(self.n, self.n), self.net).flatten()
        self.u_bar = self.u_bar + x - v
        self.x_tilde = v - self.u_bar
        self.v_tilde = x + self.u_bar
        if self.save_f:
            self.list_t.append(time.time()-self.start)
            self.list_f.append( 0.5*np.linalg.norm(self.A.dot(x) - self.proj)**2 )
        return x