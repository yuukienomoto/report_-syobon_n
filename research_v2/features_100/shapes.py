"""Exact structural search. No opponent private tiles or actual future draws.

Small bounded memoization shares hypothetical shapes across all feature rules.
Fixed melds (including ankan) disable seven-pairs and thirteen-orphans shapes.
"""
from functools import lru_cache
from itertools import combinations
from mahjong_research.tenhou import _counts, _completions, wait_shapes

TERMINALS = (0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33)


@lru_cache(maxsize=120000)
def partitions(c):
    """All nondominated (melds, taatsu, head) decompositions of concealed tiles."""
    i = next((i for i, n in enumerate(c) if n), None)
    if i is None:
        return ((0, 0, 0),)
    a = list(c)
    a[i] -= 1
    out = set(partitions(tuple(a)))  # unused single tile
    for tiles, add in (((i, i, i), (1, 0, 0)), ((i, i), (0, 1, 0)),
                       ((i, i), (0, 0, 1))):
        if c[i] >= len(tiles):
            a = list(c)
            for t in tiles: a[t] -= 1
            for m, b, p in partitions(tuple(a)):
                v = (m+add[0], b+add[1], p+add[2])
                if v[0] <= 4 and v[2] <= 1: out.add(v)
    if i < 27:
        options = []
        if i % 9 <= 6 and c[i+1] and c[i+2]: options.append(((i, i+1, i+2), (1, 0, 0)))
        for delta in (1, 2):
            if i % 9+delta <= 8 and c[i+delta]: options.append(((i, i+delta), (0, 1, 0)))
        for tiles, add in options:
            a = list(c)
            for t in tiles: a[t] -= 1
            for m, b, p in partitions(tuple(a)):
                if m+add[0] <= 4: out.add((m+add[0], b+add[1], p))
    # Preserve different head/meld trade-offs; only prune same (m,p) lower taatsu.
    best = {}
    for m, b, p in out: best[m, p] = max(b, best.get((m, p), -1))
    return tuple((m, b, p) for (m, p), b in sorted(best.items()))


@lru_cache(maxsize=60000)
def shanten(c, fixed=0):
    if len(c) != 34 or min(c) < 0 or max(c) > 4 or not 0 <= fixed <= 4:
        raise ValueError('invalid shape counts')
    standard = min(8-2*(m+fixed)-min(b, 4-m-fixed)-p
                   for m, b, p in partitions(c) if m+fixed <= 4)
    pairs = 6-sum(n >= 2 for n in c)+max(0, 7-sum(n > 0 for n in c)) if not fixed else 99
    orphans = 13-sum(c[t] > 0 for t in TERMINALS)-int(any(c[t] >= 2 for t in TERMINALS)) if not fixed else 99
    return standard, pairs, orphans


def physical(c):
    return [4*t+j for t, n in enumerate(c) for j in range(n)]


def post(c, t):
    a = list(c); a[t] -= 1
    return tuple(a)


@lru_cache(maxsize=40000)
def ukeire(c, fixed, known, owned_meld):
    """13-equivalent shape, with immutable public counts including candidate discard."""
    s = min(shanten(c, fixed))
    total = 0
    for t in range(34):
        if known[t] >= 4 or c[t]+owned_meld[t] >= 4: continue
        a = list(c); a[t] += 1; a = tuple(a)
        # A 14-equivalent partition may leave a single unused tile. Except for
        # completion, its minimum equals the best 13-equivalent discard shape.
        # The equivalence is independently checked by enumerating discards in tests.
        if min(shanten(a, fixed)) < s:
            total += 4-known[t]
    return total


@lru_cache(maxsize=6000)
def improvement(c, fixed, known, owned_meld):
    base = ukeire(c, fixed, known, owned_meld)
    s = min(shanten(c, fixed)); z = complete = 0
    for t in range(34):
        if known[t] == 4 or c[t]+owned_meld[t] == 4: continue
        a = list(c); a[t] += 1; a = tuple(a)
        if _completions(a, fixed):
            complete += 4-known[t]; continue
        k = list(known); k[t] += 1; k = tuple(k)
        values = [ukeire(post(a, d), fixed, k, owned_meld) for d, n in enumerate(a)
                  if n and min(shanten(post(a, d), fixed)) <= s]
        z += (4-known[t])*max(0, max(values, default=base)-base)
    return z, complete


def isolated(c, t):
    return c[t] == 1 and (t >= 27 or all(not c[q] for q in range(max(t//9*9,t-2),min(t//9*9+8,t+2)+1) if q != t))


def blocks(c):
    kinds = {t for t, n in enumerate(c) if n}
    pairs = [{a,b} for a,b in combinations(sorted(kinds),2) if a < 27 and a//9 == b//9 and b-a <= 2]
    overlap = sum(len(a & b) == 1 for a,b in combinations(pairs,2))
    edge = sum({s,s+1} <= kinds for s in (0,7,9,16,18,25))
    closed = sum({s,s+2} <= kinds for s in range(27) if s % 9 <= 6)
    three = any(all(sum(suit*9+r+j in kinds for j in range(3)) >= 2 for suit in range(3)) for r in range(7))
    straight = any(all(sum(suit*9+r+j in kinds for j in range(3)) >= 2 for r in (0,3,6)) for suit in range(3))
    return overlap, edge, closed, three, straight


def wall_blocked(t, known):
    paths = []
    for start in range(t//9*9, t//9*9+7):
        seq = set(range(start,start+3))
        if t not in seq or t == start+1: continue
        if (t == start+2 and start % 9 == 0) or (t == start and start % 9 == 6): continue
        paths.append(seq-{t})
    return t < 27 and bool(paths) and all(any(known[q] == 4 for q in p) for p in paths)
