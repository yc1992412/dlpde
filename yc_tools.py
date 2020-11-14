import numpy as np

def sampling(batch = 64):
    
    pi = np.pi

    batch_bd = int(np.sqrt(batch))
    
    b1 = np.zeros((batch_bd, 2))
    b2 = np.zeros((batch_bd, 2))
    b3 = np.zeros((batch_bd, 2))
    b4 = np.zeros((batch_bd, 2))
    
    b1[:,1] = 0
    b1[:,0] = np.random.rand(batch_bd)
    
    b2[:,1] = 1
    b2[:,0] = np.random.rand(batch_bd)
    
    b3[:,0] = 0
    b3[:,1] = np.random.rand(batch_bd)
    
    b4[:,0] = 1
    b4[:,1] = np.random.rand(batch_bd)
    
    bd_data = np.vstack([b1,b2,b3,b4])
    
    num_inner_point = int((batch))
    
    in_data = np.random.rand(num_inner_point,2)
    
    f_data = np.zeros((num_inner_point,1))
    f_data[:,0] = np.cos(pi*in_data[:,0])*np.cos(pi*in_data[:,1])*(2*pi**2 + 1) 
    
    return [np.float32(bd_data), np.float32(in_data), np.float32(f_data)]