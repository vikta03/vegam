"""Vectorized decoder; the original raw.decode remains the numerical control."""
import numpy as np
from .raw import CHANNELS, MAX_MOVES, MOVE_BASE, wdl


def decode(arr, context=None):
    n = len(arr)
    if not n:
        raise ValueError('empty record batch')
    version = int(arr['version'][0])
    fmt = arr['format']
    if version not in (5, 6) or np.any(arr['version'] != version) or np.any(~np.isin(fmt, [1, 2])):
        raise ValueError('unsupported/mixed record version')
    if context is None:
        if getattr(arr, 'frame_ids', None) is None:
            raise ValueError('game context required')
        context = dict(initial_rooks=np.broadcast_to(arr.initial_rooks, (n, 4)),
                       initial_stm=np.full(n, arr.initial_stm),
                       frame_ids=arr.frame_ids, total_frames=np.full(n, arr.total_frames))
    rows = np.arange(n)
    planes = np.unpackbits(arr['planes'], axis=2, bitorder='big').astype(np.float32)
    x = np.zeros((n, 64, CHANNELS), np.float32)
    x[:, :, :104] = planes.transpose(0, 2, 1)
    policy = arr['policy']
    if not np.isfinite(policy).all() or np.any((policy < 0) & (policy != -1)):
        raise ValueError('invalid policy/sentinel')
    legal = policy >= 0
    counts = legal.sum(1)
    if np.any((counts < 1) | (counts > MAX_MOVES)):
        raise ValueError('invalid legal move count')
    mass = np.maximum(policy, 0).sum(1)
    if np.any(abs(mass - 1) > 1e-4):
        raise ValueError('policy mass is not one')
    classical = fmt == 1
    for j in range(4):
        rights = arr['rights'][:, j]
        files = np.where(arr['stm'] == context['initial_stm'],
                         context['initial_rooks'][:, j], context['initial_rooks'][:, j ^ 2])
        squares = files + (0 if j < 2 else 56)
        if np.any(classical & ((rights > 1) | ((rights != 0) &
                (planes[rows, 3 if j < 2 else 9, squares] != 1)))):
            raise ValueError('ambiguous classical castling identity')
        x[rows[classical], squares[classical], 104 + j] = rights[classical]
        for f in range(8):
            x[~classical, (0 if j < 2 else 56) + f, 104 + j] = (rights[~classical] >> f) & 1
    if np.any(arr['stm'] > 1):
        raise ValueError('invalid stm')
    x[:, :, 108] = arr['stm'][:, None]
    x[:, :, 109] = arr['clock'][:, None] / 100
    old, cur = planes[:, 19, :], planes[:, 6, :]
    x[:, 40:48, 110] = (old[:, 48:56] == 1) & (cur[:, 48:56] == 0) & (old[:, 32:40] == 0) & (cur[:, 32:40] == 1)
    for h in range(8):
        x[:, :, 111 + h] = (planes[:, h*13:h*13+12, :].sum((1, 2)) > 0)[:, None]
    x[:, :, 119] = 1

    r, ids = np.nonzero(legal)
    slots = np.arange(len(ids)) - np.repeat(np.cumsum(counts) - counts, counts)
    m = MOVE_BASE[ids].copy()
    pawn = planes[r, 0, m[:, 0]] != 0
    m[(m[:, 2] == 0) & pawn & (m[:, 1] >= 56), 2] = 1
    castle = (planes[r, 5, m[:, 0]] != 0) & (planes[r, 3, m[:, 1]] != 0)
    m[castle, 3] = np.where(m[castle, 1] > m[castle, 0], 1, 2)
    moves = np.zeros((n, MAX_MOVES, 4), np.int64)
    mask = np.zeros((n, MAX_MOVES), bool)
    target = np.zeros((n, MAX_MOVES), np.float32)
    moves[r, slots] = m
    mask[r, slots] = True
    target[r, slots] = policy[r, ids] / mass[r]
    action = np.zeros((n, MAX_MOVES, 3), np.float32)
    action[:, :, 1] = 1
    action_valid = np.zeros((n, MAX_MOVES), bool)
    best, best_valid = wdl(arr['best_q'], arr['best_d'])
    if version == 6:
        outcome, outcome_valid = wdl(arr['result_q'], arr['result_d'])
        played, played_valid = wdl(arr['played_q'], arr['played_d'])
        lookup = np.full(legal.shape, -1, np.int16)
        lookup[r, ids] = slots
        # Same precedence as the writer/control decoder: best overwrites played
        # when they name the same action and both labels are available.
        for field, values, valid in [('played_idx', played, played_valid), ('best_idx', best, best_valid)]:
            idx = arr[field].astype(np.int64)
            if np.any(idx >= legal.shape[1]):
                raise ValueError(f'{field} outside policy space')
            slot = lookup[rows, idx]
            if np.any(slot < 0):
                raise ValueError(f'{field} is not a legal policy index')
            action[rows[valid], slot[valid]] = values[valid]
            action_valid[rows[valid], slot[valid]] = True
    else:
        q = arr['old_result'].astype(np.float32)
        outcome, outcome_valid = wdl(q, (q == 0).astype(np.float32))
    censored = (arr['flags'] & 0x30) != 0
    duration = (context['total_frames'] - context['frame_ids'] - censored.astype(np.int64)).astype(np.float32)
    duration_valid = duration >= 0
    outcome_valid &= (arr['flags'] & 0x10) == 0
    return dict(features=x, moves=moves, legal=mask, policy=target,
                state_wdl=best, state_valid=best_valid, outcome=outcome,
                outcome_valid=outcome_valid, action_wdl=action, action_valid=action_valid,
                duration=np.where(duration_valid, duration, 0), duration_valid=duration_valid, censored=censored)
