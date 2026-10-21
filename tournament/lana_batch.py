"""Opt-in larger inference batches for the preserved Lana checkpoint.

No neural equations or parameters change. Each <=32-row slice is validated by
its original model contract; the full tensor's leading dimensions and an explicit
new total bound are also checked. Canonical model.py is never edited.
"""
from types import MethodType
import torch

def allow_large_batch(world,limit):
 if not isinstance(limit,int) or limit<1:raise ValueError('Positive batch bound required')
 if limit<=32:return
 if world.training:raise ValueError('This adapter is for frozen evaluation only')
 from survival_agent.hybrid.model import TransformerLatent
 original=world._check_state
 def checked(this,state):
  if this.training:raise ValueError("Large-batch adapter requires evaluation mode")
  if not isinstance(state,TransformerLatent) or state.deter.ndim!=2:return original(state)
  batch=state.deter.shape[0]
  if batch<=32:return original(state)
  if batch>limit:raise ValueError('Batch exceeds adapter bound')
  tensors=[state.deter,state.stoch,state.history,state.lengths]
  if any(not torch.is_tensor(x) or x.ndim<1 or x.shape[0]!=batch for x in tensors):raise ValueError('Mismatched latent batch dimensions')
  for offset in range(0,batch,32):original(TransformerLatent(*(x[offset:offset+32] for x in tensors)))
  return batch
 world._check_state=MethodType(checked,world)
