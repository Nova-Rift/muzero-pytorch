"""Episode returns; termination kills bootstrap, truncation retains it."""
import numpy as np
def episode_returns(rewards,terminated,bootstrap,discount):
 result=np.zeros((len(rewards)+1,*rewards.shape[1:]),dtype=np.float32)
 result[-1]=bootstrap
 for t in range(len(rewards)-1,-1,-1):
  result[t]=rewards[t]+discount*(~terminated[t])*result[t+1]
 return result
