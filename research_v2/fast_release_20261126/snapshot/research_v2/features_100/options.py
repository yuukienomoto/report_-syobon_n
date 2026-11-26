"""Lazy candidate calculations shared by the complete feature registry."""
from functools import cached_property
from mahjong_research.tenhou import _counts, wait_shapes, is_suji, dora_count
from .shapes import shanten, post, ukeire, improvement, blocks, isolated, wall_blocked


class Unknown(ValueError):
    pass


def best(choices, values, maximum=True):
    if not choices: return set()
    v = (max if maximum else min)(values[d] for d in choices)
    return {d for d in choices if values[d] == v}


class Options:
    def __init__(self, state):
        self.s = state
        self.D = set(state['hand'])
        self.c = _counts(state['hand'])
        self.melds = state['melds']
        self.opp = set(state['active'])-{state['seat']}
        self.rivers = state['rivers']
        self.prev = state.get('previous')

    @cached_property
    def waits(self):
        by_kind = {}
        for d in self.D:
            if d//4 not in by_kind:
                h = list(self.s['hand']); h.remove(d)
                by_kind[d//4] = wait_shapes(h, self.melds)
        return {d: by_kind[d//4] for d in self.D}

    @cached_property
    def T(self): return {d for d in self.D if self.waits[d]}

    @cached_property
    def G(self):
        return {d for d in self.D if len(self.opp) == 1 and d//4 in self.rivers[next(iter(self.opp))]}

    @cached_property
    def Gall(self):
        return {d for d in self.D if self.opp and all(d//4 in self.rivers[p] for p in self.opp)}

    @cached_property
    def J(self):
        return {d for d in self.D if len(self.opp) == 1 and is_suji(d//4,self.rivers[next(iter(self.opp))])}

    @cached_property
    def Sforms(self):
        vals = {d: shanten(post(self.c,d//4),len(self.melds)) for d in self.D}
        if any(min(vals[d]) == 0 and not self.waits[d] for d in self.D):
            raise Unknown('own_exhausted_shape_uncertain')
        return vals

    @cached_property
    def S(self): return {d:min(v) for d,v in self.Sforms.items()}
    @cached_property
    def Dm(self): return best(self.D,self.S,False)
    @cached_property
    def Smin(self): return min(self.S.values())
    @cached_property
    def known(self): return _counts(sorted(self.s['known']))
    @cached_property
    def meld_counts(self): return _counts([t for m in self.melds for t in m.tiles])
    @cached_property
    def U(self):
        # Explicit S check preserves the formal-own-fifth-copy holdout contract.
        self.Sforms
        vals = {}
        for d in self.D:
            t = d//4
            if t not in vals:
                vals[t] = (sum(4-self.known[w] for w in self.waits[d]) if d in self.T else
                           ukeire(post(self.c,t),len(self.melds),self.known,self.meld_counts))
        return {d:vals[d//4] for d in self.D}
    @cached_property
    def Z(self):
        self.Sforms
        # Both Z rules compare only Dm (legal tenpai candidates are in Dm).
        vals = {t:improvement(post(self.c,t),len(self.melds),self.known,self.meld_counts)[0]
                for t in {d//4 for d in self.Dm}}
        return {d:vals[d//4] for d in self.Dm}
    @cached_property
    def V(self):
        total = dora_count(self.s['hand']+[t for m in self.melds for t in m.tiles], self.s['indicators'])
        return {d:total-dora_count([d],self.s['indicators']) for d in self.D}
    @cached_property
    def B(self):
        return {d:sum(t//4 in self.rivers[p] for t in self.D-{d} for p in range(4) if p != self.s['seat']) for d in self.D}
    @cached_property
    def b_before(self):
        return sum(t//4 in self.rivers[p] for t in self.D for p in range(4) if p != self.s['seat'])
    @cached_property
    def closed(self): return all(m.kind == 'ankan' for m in self.melds)
    @cached_property
    def role(self):
        # Tenhou sudden-death East game South round: East is not a round wind.
        r = self.s['round_index']//4
        return {31,32,33,27+(self.s['seat']-self.s['dealer'])%4,27+r}
    @cached_property
    def L(self):
        if not self.s.get('rules_verified'):
            raise Unknown('riichi_rule_profile_unverified')
        self.Sforms  # do not silently lose formal-tenpai exceptions
        if not self.closed or self.s['points'][self.s['seat']] < 1000 or self.s['wall'] < 4:
            return set()
        return self.T
    @cached_property
    def PF(self):
        return {d:bool(set(self.waits[d]) & (self.rivers[self.s['seat']]|{d//4})) for d in self.D}

    def pick(self, predicate, choices=None):
        return {d for d in (self.D if choices is None else choices) if predicate(d)}

    def max(self, key, choices=None):
        return best(self.Dm if choices is None else choices,getattr(self,key))

    def same_target_previous(self):
        return self.prev is not None and self.prev['free'] and self.prev['opponents'] == sorted(self.opp) and len(self.opp)==1

    def history(self):
        # Only observable information from previous OWN discard; never future state.
        record = dict(free=self.s['free'],opponents=sorted(self.opp),rivers=[sorted(r) for r in self.rivers],
                      chosen_safe=self.s['discard'] in self.G,g_empty=not self.G,
                      safe_break=bool(self.T and self.G and not self.T & self.G and self.s['discard'] in self.G),
                      r0_pass=False)
        if self.s['free'] and not self.opp:
            try: record['r0_pass'] = bool(self.L and not self.s['declaration'])
            except Unknown: record['r0_pass'] = None
        return record

    def details(self):
        out = dict(D=sorted(self.D),T=sorted(self.T),G=sorted(self.G),Gall=sorted(self.Gall),J=sorted(self.J))
        for key in ('S','U','V','Z','L'):
            if key in self.__dict__:
                v = self.__dict__[key]
                out[key] = sorted(v) if isinstance(v,set) else v
        return out
