import numpy as np 
import time
from utils import *



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
        self.u_bar = np.ones(x0.shape[0])
        
        
    def step(self, x):
        x = self.pseudo_inv.dot(self.At_proj + self.rho*self.x_tilde)
        x = np.squeeze(np.array(x))
        self.v_tilde = x + self.u_bar 
        v = denoising_image(self.v_tilde.reshape(self.n, self.n), self.net).flatten()
        self.u_bar = self.u_bar + x - v
        self.x_tilde = v - self.u_bar
        if self.save_f:
            self.list_t.append(time.time()-self.start)
            self.list_f.append( 0.5*np.linalg.norm(self.A.dot(x) - self.proj)**2 )
        return x






    
class AdaptativePlugAndPlayADMM(Optimizer):
    
    def __init__(self, x0, A, rho, proj, net, eta, gamma, save_f=False):
        super().__init__(save_f)
        self.n = int(np.sqrt(x0.shape[0]))
        self.A = A
        self.rho = rho
        self.proj = proj
        self.pseudo_inv = np.linalg.inv( (A.T).dot(A) + rho*np.eye(A.shape[1]) )
        self.At_proj = np.squeeze(np.array( (A.T).dot(proj) ))
        self.net = net
        self.eta = eta
        self.gamma = gamma
        self.x_tilde = x0.copy()
        self.v_tilde = x0.copy()
        self.v = x0.copy()
        self.u_bar = np.ones(x0.shape[0])
        self.u = self.rho * self.u_bar
        self.Delta = 0.
        
        
    def step(self, x):
        old_x = x.copy()
        # Updates
        x = self.pseudo_inv.dot(self.At_proj + self.rho*self.x_tilde)
        x = np.squeeze(np.array(x))
        self.v_tilde = x + self.u_bar 
        v = denoising_image(self.v_tilde.reshape(self.n, self.n), self.net).flatten()
        self.u_bar = self.u_bar + x - v
        u = self.rho * self.u_bar
        self.x_tilde = v - self.u_bar
        # Parameter adaptation
        Delta = 1/self.n * (np.linalg.norm(x - old_x) + np.linalg.norm(v - self.v) + np.linalg.norm(u - self.u))
        if Delta > self.eta * self.Delta:
            self.rho = self.gamma * self.rho
        # Logs
        self.Delta = Delta
        self.v = v
        self.u = u
        if self.save_f:
            self.list_t.append(time.time()-self.start)
            self.list_f.append( 0.5*np.linalg.norm(self.A.dot(x) - self.proj)**2 )
        return x




class TVOptimizer(Optimizer):

    def __init__(self, x0, A, proj, Lambda, epsilon = 1e-5, save_f=False):
        super().__init__(save_f)
        self.n = int(np.sqrt(x0.shape[0]))
        self.A = A
        self.proj = proj
        self.At_A = np.matmul(A.T, A)
        self.At_proj = (A.T).dot(proj)
        self.Lambda = Lambda
        self.epsilon = epsilon
        self.E = []
        self.rec = []
        self.TV = []
        
        
    def step(self, x):
        grad_rec = self.At_A.dot(x) - self.At_proj
        # compute the gradient of the smoothed TV functional.
        Gr = grad(x.reshape(self.n,self.n))
        Den = np.sqrt(self.epsilon**2 + np.sum(Gr**2, axis=2))
        G = -div(Gr / np.repeat(np.expand_dims(Den, axis=-1), 2, axis=-1) )
        # add the two gradients to form the update vector r
        r = grad_rec + self.Lambda*G.flatten()
        # compute step size according to energy reconstruction values
        tv_term = np.matmul(r, -div( (Gr / np.repeat(np.expand_dims(Den, axis=-1), 2, axis=-1))*grad(r.reshape(self.n,self.n)) ).flatten())
        mu = np.linalg.norm(r)**2 / ( np.linalg.norm(np.matmul(self.A, r))**2  + self.Lambda*tv_term)
        #update
        x = x - mu*r
        x = np.clip(x, 0, 1)

        if self.save_f:
            self.list_t.append(time.time()-self.start)
            self.list_f.append( 0.5*np.linalg.norm(self.A.dot(x) - self.proj)**2 )
            self.E.append( 1/2*np.linalg.norm(e.flatten())**2 + Lambda*np.sum(Den.flatten()) )
            self.rec.append( 1/2*np.linalg.norm(e.flatten())**2 )
            self.TV.append( Lambda*np.sum(Den.flatten()) )
        return x

