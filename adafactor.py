import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Adafactor:
    def __init__(self,beta,m,n,epsilon,d_model=128,warmup=200,lr=0.05):
        self.t=0
        self.tb=0
        self.e=epsilon
        self.beta= 0
        self.state={}
        
        self.m=m
        self.n=n
        self.d_model = d_model
        self.warmup = warmup
        self.lr=lr
    def get_b2(self):
        return  (1- (self.t+1)**(-0.8))
    def get_b2_b(self):
        return  (1- (self.tb+1)**(-0.8))
    def get_row_mean(self,G_sqr):
        row_mean= torch.mean(G_sqr,dim=-1,keepdim=True)
        return row_mean
    def get_lr(self):
        t = max(self.t, 1)
        return  min(t ** -0.5, self.lr)
    def get_col_mean(self,G_sqr):
        col_mean= torch.mean(G_sqr,dim=0,keepdim=True)
        return col_mean
    def rms(self, x):
        return torch.sqrt(torch.mean(x**2) + 1e-30)
    def grad(self,G,W):
        
        if(len(G.shape)==2):
            
            if('r_t' not in  self.state.keys()):
                self.state['r_t']= torch.zeros(self.m,1,device=device)
                self.state['c_t']= torch.zeros(1,self.n,device=device)
        else:
            if('g_sqr' not in  self.state.keys()):
                self.state['g_sqr']= torch.zeros(self.m,device=device)
                
        G_sqr= G*G
        # print(len(G.shape))
        # self.beta= self.get_b2()
        if(len(G.shape)==2) :
            beta= self.get_b2()
            self.state['r_t']= beta*self.state['r_t'] + (1-beta)*self.get_row_mean(G_sqr)
            self.state['c_t'] = beta*self.state['c_t'] + (1-beta)*self.get_col_mean(G_sqr)
            v_t= self.state['r_t']*(self.state['c_t'])
            v_t= v_t/torch.mean(self.state['r_t'])
            self.t+=1
        else:
            # print(G.shape)
            beta= self.get_b2_b()
            # print(self.state['g_sqr'].shape)
            self.state['g_sqr']= self.beta*self.state['g_sqr']+ (1-self.beta)*G_sqr
            v_t= self.state['g_sqr']
            self.tb+=1
        v_t = torch.clamp(v_t, min=1e-30)
        denom = torch.sqrt(v_t) + self.e
        

        u_t= G/denom

        u_t= u_t/max(1,self.rms(u_t))

        
        
        lr = self.get_lr()
        param_scale = max(1e-3, self.rms(W))
        # print(self.t)
        
        return  lr*param_scale*u_t
      
    def grad_emb(self,G,rows,W):

        if(len(G.shape)==2):
            if('r_t' not in  self.state.keys()):
                self.state['r_t']= torch.zeros(self.m,1,device=device)
                self.state['c_t']= torch.zeros(1,self.n,device=device)
        else:
            if('g_sqr' not in self.state.keys()):
                self.state['g_sqr']= torch.zeros(self.m,device=device)
        # self.beta= self.get_b2()
        G_sqr= G*G
        if (len(G.shape)==2):
            self.state['r_t'][rows]= self.beta*self.state['r_t'][rows] + (1-self.beta)*self.get_row_mean(G_sqr)
            self.state['c_t'] = self.beta*self.state['c_t'] + (1-self.beta)*self.get_col_mean(G_sqr)
            v_t= self.state['r_t'][rows]*(self.state['c_t'])
            v_t= v_t/torch.mean(self.state['r_t'][rows])
        else:
            self.state['g_sqr']= self.beta*self.state['g_sqr']+ (1-self.beta)*G_sqr
            v_t= self.state['g_sqr']
        v_t = torch.clamp(v_t, min=1e-30)
        u_t= G/(torch.sqrt(v_t)+self.e)
        u_t= u_t/max(1,self.rms(u_t))
        self.t+=1
        lr= self.get_lr()
        param_scale = max(1e-3, self.rms(W))
        return lr*param_scale*u_t
