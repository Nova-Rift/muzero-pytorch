"""Read-only intended-arm contract check; run on each restored full-state milestone."""
import argparse,json
from pathlib import Path

def check(manifest,state,contract):
    if contract not in ('baseline','root5-mass-v1'): raise ValueError('Unknown intended objective')
    if manifest['arguments']['loss_contract']!=contract: raise ValueError('Manifest objective differs from intended arm')
    if state['arguments']['loss_contract']!=contract: raise ValueError('Checkpoint objective differs from intended arm')
    if state['arguments']!=manifest['arguments']: raise ValueError('Checkpoint arguments mismatch')
    return True

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--expected-loss-contract',choices=['baseline','root5-mass-v1'],required=True);a=p.parse_args()
    import torch
    check(json.loads(a.manifest.read_text()),torch.load(a.checkpoint,map_location='cpu',weights_only=True),a.expected_loss_contract)
    print(json.dumps({'verified':True,'expected_loss_contract':a.expected_loss_contract,'scope':'Objective binding only; run alongside full-state integrity verifier'}))
