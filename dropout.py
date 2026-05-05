import torch
class Dropout:
    def __init__(self, p=0.1):
        self.p = p
        self.training = True
        self.masks = []  # stack instead of single mask

    def forward(self, x):
        if not self.training or self.p == 0:
            self.masks.append(None)
            return x
        mask = (torch.rand_like(x) > self.p).float()
        # print(x.shape,(x==0).sum())
        self.masks.append(mask)
        return x * mask / (1 - self.p)

    def backward(self, grad_out):
        
        mask = self.masks.pop()
          # LIFO matches reverse-layer backprop order
        if mask is None:
            return grad_out
        return grad_out * mask / (1 - self.p)

    def clear(self):
        self.masks.clear()

    def train(self,is_training):
        self.training = is_training
        return self