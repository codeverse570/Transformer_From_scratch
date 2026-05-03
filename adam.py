import math

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SchedulerState:
    def __init__(self, base_lr, warmup_steps=4000, total_steps=100000):
        self.global_step = 0
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def advance(self):
        self.global_step += 1

    def get_lr(self):
        step = max(self.global_step, 1)
        if step < self.warmup_steps:
            scale = step / self.warmup_steps
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            scale = max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return self.base_lr * scale
class AdamCustom:
    def __init__(self,_, m, n,__, lr=1e-4, beta1=0.9, beta2=0.999, eps=1e-8,scale=False,schedular=None):
        self.m = m
        self.n = n
        self.scale=scale
        self.schedular= schedular
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.t = 0
        self.tb=0
        self.state = {}

    def rms(self, x):
        return torch.sqrt(torch.mean(x**2) + 1e-30)

    def grad(self, G, W):
        # init state
        
    #    norm = G.norm(2)
       
    #    if norm > 1.0:
    #             G = G * (1.0 / (norm + 1e-6))
       step=0
       if (len(G.shape)==2):
        if 'm_t' not in self.state:
            self.state['m_t'] = torch.zeros(self.m,self.n, device=device)
            self.state['v_t'] = torch.zeros(self.m,self.n, device=device)
        self.t += 1
        step=self.t
        m_t=self.state['m_t']
        v_t= self.state['v_t']
       else:
        if 'm_t_b' not in self.state:
            self.state['m_t_b'] = torch.zeros(self.m, device=device)
            self.state['v_t_b'] = torch.zeros(self.m, device=device)
        self.tb += 1
        step= self.tb
        m_t=self.state['m_t_b']
        v_t= self.state['v_t_b']
            # moments
       m_t = self.beta1 * m_t + (1 - self.beta1) * G
       v_t = self.beta2 * v_t + (1 - self.beta2) * (G * G)

            # bias correction
       m_hat = m_t / (1 - self.beta1 ** step)
       v_hat = v_t / (1 - self.beta2 ** step)

            # update
       update = m_hat / (torch.sqrt(v_hat) + self.eps)
    #    if (self.scale==True):
    #       update = update / (update.norm(dim=1, keepdim=True) + 1e-6)
            # optional param scaling (to match your Adafactor behavior)
            # param_scale = max(1e-3, self.rms(W))
       if (len(G.shape)==2):
        self.state['m_t'] = m_t
        self.state['v_t'] = v_t
       else:
        self.state['m_t_b'] = m_t
        self.state['v_t_b'] = v_t

       return self.schedular.get_lr()  * update
       
            
    def grad_emb(self, G, rows, W):
        if 'm_t' not in self.state:
            self.state['m_t'] = torch.zeros(self.m,self.n,device=device)
            self.state['v_t'] = torch.zeros(self.m,self.n,device=device)
        if 't_rows' not in self.state:
            self.state['t_rows'] = torch.zeros(self.m, device=device)    

        # self.t += 1
        self.state['t_rows'][rows] += 1
        t_rows = self.state['t_rows'][rows]
        m_t = self.state['m_t']
        v_t = self.state['v_t']

        # update only selected rows
        m_t[rows] = self.beta1 * m_t[rows] + (1 - self.beta1) * G
        v_t[rows] = self.beta2 * v_t[rows] + (1 - self.beta2) * (G * G)

        m_hat = m_t[rows] / (1 - self.beta1 ** t_rows).unsqueeze(1)
        v_hat = v_t[rows] / (1 - self.beta2 ** t_rows).unsqueeze(1)

        update = m_hat / (torch.sqrt(v_hat) + self.eps)
        update = update 
        # param_scale = max(1e-3, self.rms(W))

        self.state['m_t'] = m_t
        self.state['v_t'] = v_t
        # print(self.lr)
        return self.schedular.get_lr()  *update
