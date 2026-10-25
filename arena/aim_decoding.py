"""Model action index -> arena command, mirroring arena/bridge.mjs exactly.

The model always emits one of 153 indices: move = index % 9 (3x3 grid, 4 = stay),
aim = index // 9 (0 = do not fire, 1..16 = 22.5 degree bins).  Two decodings exist:

* absolute (historical, default): angle = (aim - 1) * 22.5 degrees measured in arena
  coordinates (+x east, +y south).  Bin 1 fires due east from every seat.
* relative (egocentric): angle = bearing(self -> nearest live opponent) + (aim - 1) * 22.5
  degrees.  Bin 1 fires directly at the opponent from every seat; the move component stays
  absolute.  Bearing uses Euclidean pixel distance; ties keep the lowest fighter id; with no
  live opponent the bearing is 0, i.e. the absolute decoding.

Wire protocol: the bridge receives int32 actions.  0..152 are decoded absolutely (byte-identical
to the pre-change bridge); 153..305 are the same indices decoded relatively.  The environment,
observation and rewards are unchanged.

Observation channels used for the bearing mirror (lana/observation.ts, 64 floats per agent):
0 self.x/1000, 1 self.y/460, 2 self.hp/3, 3 cooldown/0.6; then for each of the ten other
fighters in id order (index 4 + 4*j): dx/1000, dy/460, other.hp/3, alive flag (hp > 0);
44..63 four nearest enemy shots (dx/1000, dy/460, vx/500, vy/500, present).  The JS bridge
computes the same bearing from world state (fighter x, y, hp); this module reproduces it from
the observation of the same instant.
"""
import math

ACTIONS = 153
MODES = ('absolute', 'relative')
WIRE_OFFSET = {'absolute': 0, 'relative': ACTIONS}
WIRE_ACTIONS = 2 * ACTIONS


def check_mode(mode):
    if mode not in MODES:
        raise ValueError('Invalid aim decoding mode: ' + repr(mode))
    return mode


def wire_offset(mode):
    """Constant added to the model index before it is sent to the bridge."""
    return WIRE_OFFSET[check_mode(mode)]


def wire_action(action, mode):
    if not 0 <= int(action) < ACTIONS:
        raise ValueError('Model action out of range')
    return int(action) + wire_offset(mode)


def bearing_from_observation(observation):
    """Radians from self to the nearest live other fighter, from one 64-float observation."""
    best = None
    best_distance = math.inf
    for j in range(10):
        dx, dy, _hp, alive = observation[4 + 4 * j:8 + 4 * j]
        if alive <= 0:
            continue
        distance = math.hypot(dx * 1000, dy * 460)
        if distance < best_distance:
            best_distance = distance
            best = (dx * 1000, dy * 460)
    return math.atan2(best[1], best[0]) if best else 0.0


def decode(action, mode='absolute', observation=None):
    """Return the arena command dict {dx, dy, fire, aimX, aimY} for one model index."""
    check_mode(mode)
    action = int(action)
    if not 0 <= action < ACTIONS:
        raise ValueError('Model action out of range')
    move, aim = action % 9, action // 9
    angle = (aim - 1) * math.pi / 8
    if mode == 'relative' and aim > 0:
        if observation is None:
            raise ValueError('Relative decoding needs the observation')
        angle += bearing_from_observation(observation)
    return dict(dx=move % 3 - 1, dy=move // 3 - 1, fire=aim > 0, aimX=math.cos(angle), aimY=math.sin(angle))
