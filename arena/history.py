"""Per-agent history on the model device: transfer each observation only once."""
import torch

class DeviceHistory:
    def __init__(self,count,context,device,observations=64,actions=153):
        self.obs=torch.zeros(count,context,observations,device=device)
        self.actions=torch.zeros(count,context-1,device=device)
        self.action_count=actions

    def observe(self,observations):
        self.obs=torch.roll(self.obs,-1,1)
        self.obs[:,-1].copy_(torch.as_tensor(observations,device=self.obs.device))
        return torch.cat((self.obs.flatten(1),self.actions),-1)

    def acted(self,actions):
        if self.actions.shape[1]:
            self.actions=torch.roll(self.actions,-1,1)
            self.actions[:,-1].copy_(torch.as_tensor(actions,device=self.obs.device)/self.action_count)
