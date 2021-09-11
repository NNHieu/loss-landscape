import torch 
import torch.nn as nn 
import torch.nn.functional as F

import torch.autograd as autograd 

from torchvision import datasets, transforms 
from torch.utils.data import DataLoader 

import torch.optim as optim

class ResNetLayer(nn.Module): 
    def __init__(self, n_channels, n_inner_channels, kernel_size=3, num_groups=8): 
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, n_inner_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.conv2 = nn.Conv2d(n_inner_channels, n_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.norm1 = nn.GroupNorm(num_groups, n_inner_channels)
        self.norm2 = nn.GroupNorm(num_groups, n_channels)
        self.norm3 = nn.GroupNorm(num_groups, n_channels) 
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        
    def forward(self, z, x):
        y = self.norm1(F.relu(self.conv1(z)))
        return self.norm3(F.relu(z + self.norm2(x + self.conv2(y))))

def anderson(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta = 1.0): 
    """ Anderson acceleration for fixed point iteration. """ 
    bsz, d, H, W = x0.shape 
    X = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device) 
    F = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
    X[:,0], F[:,0] = x0.view(bsz, -1), f(x0).view(bsz, -1) 
    X[:,1], F[:,1] = F[:,0], f(F[:,0].view_as(x0)).view(bsz, -1) 
    H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device) 
    H[:,0,1:] = H[:,1:,0] = 1 
    y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device) 
    y[:,0] = 1 
    res = [] 
    for k in range(2, max_iter): 
        n = min(k, m) 
        G = F[:,:n]-X[:,:n] 
        H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None] 
        alpha = torch.linalg.solve(H[:,:n+1,:n+1], y[:,:n+1])[:, 1:n+1, 0] # (bsz x n) 
        X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0] 
        F[:,k%m] = f(X[:,k%m].view_as(x0)).view(bsz, -1) 
        res.append((F[:,k%m] - X[:,k%m]).norm().item()/(1e-5 + F[:,k%m].norm().item())) 
        if (res[-1] < tol): break 
    return X[:,k%m].view_as(x0), res

class DEQFixedPoint(nn.Module): 
    def __init__(self, f, solver, **kwargs): 
        super().__init__() 
        self.f = f 
        self.solver = solver 
        self.kwargs = kwargs 
        
    def forward(self, x): 
        # compute forward pass and re-engage autograd tape 
        with torch.no_grad(): 
            z, self.forward_res = self.solver(lambda z : self.f(z, x), torch.zeros_like(x), **self.kwargs) 
        z = self.f(z,x) 
            
        # set up Jacobian vector product (without additional forward calls) 
        # z0 = z.clone().detach().requires_grad_() 
        # f0 = self.f(z0,x) 
    
        # def backward_hook(grad): 
        #     g, self.backward_res = self.solver(lambda y : autograd.grad(f0, z0, y, retain_graph=True)[0] + grad, grad, **self.kwargs) 
        #     return g 
    
        # z.register_hook(backward_hook) 
        return z

def SimpleDEQ():
    chan = 48
    f = ResNetLayer(chan, 64, kernel_size=3) 
    model = nn.Sequential(nn.Conv2d(3,chan, kernel_size=3, bias=True, padding=1), 
                        nn.BatchNorm2d(chan), 
                        DEQFixedPoint(f, anderson, tol=1e-2, max_iter=25, m=5), 
                        nn.BatchNorm2d(chan), 
                        nn.AvgPool2d(8,8), 
                        nn.Flatten(), 
                        nn.Linear(chan*4*4,10))
    return model