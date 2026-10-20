"""Standalone real-interaction cue-task smoke run; no external environment needed."""
import argparse
import json
import torch
from model import Config, MuZero, search


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--updates',type=int,default=10)
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=parser.parse_args()
    torch.manual_seed(7);torch.set_num_threads(1)
    model=MuZero(Config(observations=2,actions=2,hidden=32,simulations=12)).to(args.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    for update in range(args.updates):
        cues=torch.randint(2,(8,),device=args.device)
        observations=torch.nn.functional.one_hot(cues,2).float()
        policies=torch.stack([search(model,o,explore=True)[0] for o in observations])
        actions=torch.distributions.Categorical(probs=policies).sample()
        rewards=(actions==cues).float()[:,None]
        values=torch.cat((rewards,torch.zeros_like(rewards)),1)
        targets=torch.stack((policies,torch.ones_like(policies)/2),1)
        valid=torch.ones_like(values);valid[:,1]=0 # Terminal state has no policy decision.
        loss=model.loss(observations,actions[:,None],rewards,values,targets,valid).mean()
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),100.,error_if_nonfinite=True);optimizer.step()
        print(json.dumps(dict(update=update+1,loss=float(loss.detach()),device=args.device)))

if __name__=='__main__':main()
