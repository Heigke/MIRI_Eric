import numpy as np
import scipy
from scipy import linalg
import matplotlib

class Lqr:
    
    def __init__(self,A,B,Q,R,N,nbr_steps_future):
        
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.N = N
        self.nbr_steps_future = nbr_steps_future
        self.count_steps = nbr_steps_future+2
        self.Fvec = []
        return 
        
    def CalcP(self,nbr_steps):
        P_array = []
        
        A = self.A
        B = self.B
        Q = self.Q
        R = self.R
        N = self.N
        
        P_array.insert(0,Q)
        
        for i in range(nbr_steps-1):
            
            big_thing = (np.transpose(A)@P_array[0]@B+N)@linalg.inv(R+np.transpose(B)@P_array[0]@B)
            
            more_big_thing = big_thing@(np.transpose(B)@P_array[0]@A+np.transpose(N))
            
            P_next = np.transpose(A)@(P_array[0]@A)-more_big_thing+Q
            P_array.insert(0,P_next)
        return P_array
    def f_vec(self,p_array):
        
        A = self.A
        B = self.B
        Q = self.Q
        R = self.R
        N = self.N
        
        F_vec = []
        for i in range(len(p_array)-1):
            
            F_k = linalg.inv(R+np.transpose(B)@p_array[i+1]@B)@(np.transpose(B)@p_array[i+1]@A+np.transpose(N))
            F_vec.append(F_k)
            
        return F_vec
       
    
    def reset(self):
        self.count_steps=0
        P_array = self.CalcP(self.nbr_steps_future)
        self.F_vec = self.f_vec(P_array)
        
    def action(self,x_0):
        
        if self.count_steps >= self.nbr_steps_future-1:
            self.count_steps=0
            P_array = self.CalcP(self.nbr_steps_future)
            self.F_vec = self.f_vec(P_array)
        
        self.count_steps += 1
        return -self.F_vec[self.count_steps-1]@x_0

            
            
            
            
