"""Restore a hash-bound full state outside the immutable trainer tree."""
import argparse,hashlib,importlib.util,json,platform,time
from pathlib import Path

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while b:=f.read(1048576):h.update(b)
 return h.hexdigest()

def restore(source,baseline,compressed,output,binding,torch_version,python_version):
 source,baseline,compressed,output=map(Path,(source,baseline,compressed,output))
 manifest=json.loads((baseline/'manifest.json').read_text());provenance=manifest['source_provenance']
 if torch_version!=manifest['torch'] or python_version!=manifest['python']:raise ValueError('Runtime differs from original full state')
 if provenance['dirty'] or provenance['commit']!=binding['trainer_commit']:raise ValueError('Source provenance differs')
 receipt=json.loads((source/'.source-provenance.json').read_text())
 if receipt['commit']!=provenance['commit'] or receipt['dirty'] or receipt['files']!=provenance['files']:raise ValueError('Source receipt differs')
 for name,sha in provenance['files'].items():
  p=(source/name).resolve()
  if not p.is_relative_to(source.resolve()) or digest(p)!=sha:raise ValueError('Frozen source differs: '+name)
 for name,sha in binding['baseline_files'].items():
  p=(baseline/name).resolve()
  if not p.is_relative_to(baseline.resolve()) or digest(p)!=sha:raise ValueError('Baseline binding differs')
 if compressed.stat().st_size!=binding['compressed_bytes'] or digest(compressed)!=binding['compressed_sha256']:raise ValueError('Compressed state differs')
 output.mkdir(parents=True,exist_ok=False)
 spec=importlib.util.spec_from_file_location('frozen_compression',source/'diagnostics/retention/compressed_checkpoint.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 raw=output/'resume-state.pt';start=time.monotonic();mod.restore(compressed,raw,binding['raw_sha256'],binding['raw_bytes'])
 result=dict(restored=True,raw_sha256=binding['raw_sha256'],raw_bytes=binding['raw_bytes'],source_commit=binding['trainer_commit'],torch=torch_version,python=python_version,elapsed_seconds=time.monotonic()-start,scope='Hash-bound bytes/runtime/source admission; trainer enforces optimizer/replay/RNG/config restoration')
 (output/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--baseline',required=True);p.add_argument('--compressed',required=True);p.add_argument('--output',required=True);p.add_argument('--binding',required=True);a=p.parse_args()
 import torch
 print(json.dumps(restore(a.source,a.baseline,a.compressed,a.output,json.loads(Path(a.binding).read_text()),str(torch.__version__),platform.python_version())))
if __name__=='__main__':main()
