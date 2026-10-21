"""Source evidence for local Git checkouts and private cached snapshots."""
import hashlib,json,subprocess
from pathlib import Path

def capture_source(root):
 root=Path(root).resolve();receipt=root/'.source-provenance.json'
 if receipt.exists():
  data=json.loads(receipt.read_text())
  for name,expected in data['files'].items():
   path=(root/name).resolve()
   if not path.is_relative_to(root):raise ValueError('Invalid source receipt path')
   if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Cached source changed: '+name)
  return {'commit':data['commit'],'dirty':data['dirty'],'files':data['files'],'kind':'verified-cache-receipt'}
 # Require the exact repository root: do not accidentally identify a parent repo.
 actual=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=root,text=True).strip()).resolve()
 if actual!=root:raise ValueError('Source root is not its own Git repository')
 tracked=subprocess.check_output(['git','ls-files','-z'],cwd=root).decode().split('\0')
 return {'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
         'dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()),
         'files':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in tracked if n},'kind':'local-git-files'}
