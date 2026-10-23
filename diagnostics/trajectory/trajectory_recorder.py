"""Bounded observer for an unchanged arena Pool; never selects or rewrites actions."""
import json
from pathlib import Path

class RecordingPool:
    def __init__(self,inner,output,max_games=8,max_steps=256):
        if type(max_games) is not int or not 1<=max_games<=128:raise ValueError('max_games')
        if type(max_steps) is not int or not 1<=max_steps<=256:raise ValueError('max_steps')
        self.inner=inner;self.max_games=max_games;self.max_steps=max_steps
        self.file=Path(output).open('x');self.steps=0;self.reset_done=False
    def emit(self,kind,state,actions=None):
        # Serialization creates owned values; no state/action references retained.
        row=dict(kind=kind,step=self.steps,seeds=self.seeds)
        for key in ('observation','alive','reward','finished','winner'):
            row[key]=state[key][:self.max_games].tolist()
        if actions is not None:row['actions']=actions[:self.max_games].tolist()
        self.file.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n');self.file.flush()
    def reset(self,seeds):
        if self.reset_done:raise RuntimeError('One reset per trace file')
        seeds=list(seeds);self.seeds=[int(x) for x in seeds[:self.max_games]]
        state=self.inner.reset(seeds);self.reset_done=True;self.emit('reset',state);return state
    def step(self,actions):
        if not self.reset_done:raise RuntimeError('Reset required')
        # Capture actual submitted actions before the backend may reuse buffers.
        captured=actions[:self.max_games].copy() if self.steps<self.max_steps else None
        state=self.inner.step(actions);self.steps+=1
        if captured is not None:self.emit('transition',state,captured)
        return state
    def close(self):
        try:self.inner.close()
        finally:self.file.close()
