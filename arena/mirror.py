"""Training-only horizontal (east-west) mirror of arena observations and actions.

Mirror axis: the arena centre x = 500 (normalised 0.5), NOT the spawn midline x = 260.
Walls (22..978), shot retention (0..1000) and the dead-fighter obstacle layout
(x = 180, 340, 500, 660, 820) are exactly symmetric about x = 500, so every mirrored
state is physically realisable under the unchanged rules; mirroring about x = 260 would
place fighters outside the arena. Spawns (seat 0 at x = 180, seat 1 at x = 340) are not
symmetric about 500, so mirrored episodes look like games spawned on the east half:
off-distribution in absolute x, which is exactly the coordinate the policy must stop using.

Observation layout (lana/observation.ts 1867ab29, 64 floats):
  0 self.x/1000      1 self.y/460    2 self.hp/3    3 cooldown/.6
  4+4k .. 7+4k, k = 0..9 (other fighters by id, self excluded): dx/1000, dy/460, hp/3, alive
  44+5s .. 48+5s, s = 0..3 (nearest enemy shots by distance): dx/1000, dy/460, vx/500, vy/500, present
Mirror: x -> 1 - x on channel 0; negate every dx and vx channel; everything else unchanged.
Shot slots are ordered by distance, which the mirror preserves, so slot order is kept.
Fighter slots are kept by id: dead fighters never move, so their dx channels are exact
mirrors geometrically but read as positions that id never occupies (documented limitation).

Action layout (arena/bridge.mjs): a = aim*9 + move, move = a % 9 with dx = move%3 - 1,
dy = move//3 - 1; aim 0 = no fire, aim k >= 1 fires at angle (k-1)*pi/8 (k = 1 east, 5 south
in screen coordinates, 9 west, 13 north). Mirror: dx -> -dx, angle -> pi - angle, i.e.
k -> ((9 - k) mod 16) + 1 for k >= 1 and aim 0 stays 0.
"""
import numpy as np

NEGATE=np.array([4+4*k for k in range(10)]+[44+5*s for s in range(4)]+[46+5*s for s in range(4)])
ONE_MINUS=np.array([0])


def mirror_aim(aim):
    aim=np.asarray(aim);return np.where(aim>0,(9-aim)%16+1,0)


def mirror_move(move):
    move=np.asarray(move);return (move//3)*3+(2-move%3)


def action_permutation():
    a=np.arange(153);return mirror_aim(a//9)*9+mirror_move(a%9)


ACTION=action_permutation()


def mirror_observation(obs):
    """obs: (..., 64) float array of raw observations (no padding rows). Returns a mirrored copy."""
    out=np.array(obs,copy=True)
    out[...,ONE_MINUS]=1-out[...,ONE_MINUS]
    out[...,NEGATE]=-out[...,NEGATE]
    return out


def mirror_policy(policy):
    """policy: (..., 153); mirrored policy puts mass of action a on ACTION[a] (ACTION is an involution)."""
    return np.asarray(policy)[...,ACTION]
