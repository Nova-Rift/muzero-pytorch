"""Exact recursive checkpoint comparison; no implicit ignored fields or tolerances."""
import argparse,json
import torch
import numpy as np

def compare(a,b,path="state"):
 if type(a) is not type(b):return {"path":path,"reason":"type"}
 if isinstance(a,torch.Tensor):
  if a.dtype!=b.dtype or a.shape!=b.shape:return {"path":path,"reason":"tensor shape/dtype"}
  if not torch.equal(a,b):return {"path":path,"reason":"tensor value"}
 elif isinstance(a,np.ndarray):
  if a.dtype!=b.dtype or a.shape!=b.shape or not np.array_equal(a,b):return {"path":path,"reason":"array"}
 elif isinstance(a,dict):
  if a.keys()!=b.keys():return {"path":path,"reason":"keys"}
  for k in a:
   error=compare(a[k],b[k],f"{path}.{k}")
   if error:return error
 elif isinstance(a,(list,tuple)):
  if len(a)!=len(b):return {"path":path,"reason":"length"}
  for i,(x,y) in enumerate(zip(a,b)):
   error=compare(x,y,f"{path}[{i}]")
   if error:return error
 elif a!=b:return {"path":path,"reason":"scalar"}
 return None

if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("left");p.add_argument("right");p.add_argument("--output",required=True);a=p.parse_args()
 x=torch.load(a.left,map_location="cpu",weights_only=True);y=torch.load(a.right,map_location="cpu",weights_only=True)
 mismatch=compare(x,y)
 with open(a.output,"w") as f:json.dump({"exact":mismatch is None,"first_mismatch":mismatch,"ignored_fields":[]},f,indent=2)
 raise SystemExit(0 if mismatch is None else 1)
