import math


class WarmupScheduler:

    def __init__(self, optimizer, warmup_steps, total_steps, max_lr, min_lr=0.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
    
    def step(self):

        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            lr = self.max_lr * (self.current_step / self.warmup_steps)
        else:
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
