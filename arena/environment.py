"""Independent Node processes, binary float32 observations, canonical JS rules."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import struct
import subprocess
import numpy as np

STRIDE=729

class Pool:
    def __init__(self,site,workers=1,node='node'):
        self.pool=ThreadPoolExecutor(max_workers=workers)
        self.processes=[subprocess.Popen([node,'--experimental-strip-types',str(Path(__file__).with_name('bridge.mjs')),str(Path(site).resolve())],stdin=subprocess.PIPE,stdout=subprocess.PIPE) for _ in range(workers)]

    @staticmethod
    def read(stream,size):
        parts=[]
        while size:
            part=stream.read(size)
            if not part:raise RuntimeError('Arena worker exited')
            parts.append(part);size-=len(part)
        return b''.join(parts)

    @classmethod
    def exchange(cls,process,payload):
        process.stdin.write(struct.pack('<I',len(payload))+payload);process.stdin.flush()
        size=struct.unpack('<I',cls.read(process.stdout,4))[0]
        rows=np.frombuffer(cls.read(process.stdout,size),dtype='<f4').reshape(-1,STRIDE).copy()
        return rows

    def reset(self,seeds):
        self.splits=np.array_split(np.asarray(seeds,dtype='<i4'),len(self.processes))
        if any(len(s)==0 for s in self.splits):raise ValueError('Need at least one game per worker')
        return self.send([struct.pack('<ii',0,len(s))+s.tobytes() for s in self.splits])

    def step(self,actions):
        splits=np.split(np.asarray(actions,dtype='<i4'),np.cumsum([len(s) for s in self.splits])[:-1])
        return self.send([struct.pack('<ii',1,len(s))+s.tobytes() for s in splits])

    def send(self,payloads):
        futures=[self.pool.submit(self.exchange,p,b) for p,b in zip(self.processes,payloads)]
        rows=np.concatenate([f.result() for f in futures])
        return dict(observation=rows[:,:704].reshape(-1,11,64),alive=rows[:,704:715].astype(bool),reward=rows[:,715:726],finished=rows[:,726].astype(bool),winner=rows[:,727].astype(int))

    def close(self):
        for process in self.processes:
            process.stdin.close()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=10)
            process.stdout.close()
        self.pool.shutdown()
