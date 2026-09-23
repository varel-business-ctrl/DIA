from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from collections import Counter
from typing import FrozenSet, Optional

WIDTH=10; HEIGHT=10; MAX_ACTIONS=160
Cell=tuple[int,int]; Offset=tuple[int,int]

class Mode(str,Enum):
    NO_MEMORY='no_memory'; EPISODIC_MEMORY='episodic_memory'; LEARNED_RELATION='learned_relation'; LEARNED_ABLATION='learned_ablation'
@dataclass(frozen=True)
class Task: name:str; start:Cell; target:Cell; obstacles:FrozenSet[Cell]
@dataclass
class TrialResult:
    task:str; mode:str; solved:bool; actions:int; failures:int; prior_cells:int; influence_events:int; route_changes:int; learned_observations:int; used_patterns:int

def add(a,b): return (a[0]+b[0],a[1]+b[1])
def sub(a,b): return (a[0]-b[0],a[1]-b[1])
def inside(c): return 1<=c[0]<=WIDTH and 1<=c[1]<=HEIGHT
def neighbours(c): return [(c[0]+1,c[1]),(c[0]-1,c[1]),(c[0],c[1]+1),(c[0],c[1]-1)]
def bfs(start,target,blocked):
    if start in blocked or target in blocked:return None
    q=[start]; parent={start:None}
    for cur in q:
        if cur==target:
            out=[]; n=cur
            while n is not None: out.append(n); n=parent[n]
            return out[::-1]
        for nxt in neighbours(cur):
            if inside(nxt) and nxt not in blocked and nxt not in parent:
                parent[nxt]=cur; q.append(nxt)
    return None

def transform(o,rotation,reflection):
    x,y=o
    if reflection:x=-x
    for _ in range(rotation%4): x,y=-y,x
    return x,y

def canonicalize(o): return min(transform(o,r,f) for f in (False,True) for r in range(4))

class LatentRelationMemory:
    MIN_SUPPORT=2
    def __init__(self): self.counts=Counter(); self.total_observations=0

    def _frame(self, src, target, blocked):
        """Express the blocked-cell offset in a target-aligned frame."""
        dx,dy=sub(target,src)
        ox,oy=sub(blocked,src)
        if abs(dx)>=abs(dy):
            # Target is primarily east/west. Make forward = +x.
            if dx<0: ox=-ox
            return ox,oy
        # Target is primarily north/south. Make forward = +x.
        if dy<0: ox,oy=oy,-ox
        else: ox,oy=-oy,ox
        return ox,oy

    def observe_failed_move(self,src,target,blocked):
        self.counts[self._frame(src,target,blocked)]+=1
        self.total_observations+=1

    def usable_patterns(self): return {p for p,c in self.counts.items() if c>=self.MIN_SUPPORT}

    def predict(self,current,target):
        """Re-project learned target-frame relations into world coordinates."""
        dx,dy=sub(target,current)
        if abs(dx)>=abs(dy):
            forward=(1,0) if dx>=0 else (-1,0)
            side=(0,1)
        else:
            forward=(0,1) if dy>=0 else (0,-1)
            side=(1,0)
        out=set()
        for fx,sy in self.usable_patterns():
            c=(current[0]+fx*forward[0]+sy*side[0], current[1]+fx*forward[1]+sy*side[1])
            if inside(c): out.add(c)
        return out


class AgentMemory:
    def __init__(self): self.episodic=set(); self.latent=LatentRelationMemory()
    def observe_failure(self,src,target,blocked): self.episodic.add(blocked); self.latent.observe_failed_move(src,target,blocked)

def execute(task,mode,memory,learn=True):
    pos=task.start; failures=actions=influence=changes=used=0
    prior_epi=set(memory.episodic) if memory and mode==Mode.EPISODIC_MEMORY else set()
    prior_pred=set(memory.latent.predict(pos,task.target)) if memory and mode==Mode.LEARNED_RELATION else set()
    local_failed=set()
    if prior_pred: used=len(memory.latent.usable_patterns())
    while pos!=task.target and actions<MAX_ACTIONS:
        known=set()
        if mode==Mode.EPISODIC_MEMORY: known|=prior_epi
        if mode==Mode.LEARNED_RELATION: known|=prior_pred
        known |= local_failed
        base=bfs(pos,task.target,set()); constrained=bfs(pos,task.target,known)
        if base!=constrained: influence+=1; changes+=1
        route=constrained if constrained is not None else base
        if route is None or len(route)<2: break
        nxt=route[1]; actions+=1
        if nxt in task.obstacles:
            failures+=1
            local_failed.add(nxt)
            if memory and learn and mode!=Mode.LEARNED_ABLATION: memory.observe_failure(pos,task.target,nxt)
            if mode==Mode.EPISODIC_MEMORY: prior_epi.add(nxt)
            elif mode==Mode.LEARNED_RELATION:
                prior_pred=memory.latent.predict(pos,task.target); prior_pred.add(nxt); prior_pred.discard(pos)
        else: pos=nxt
    return TrialResult(task.name,mode.value,pos==task.target,actions,failures,len(prior_epi if mode==Mode.EPISODIC_MEMORY else prior_pred),influence,changes,memory.latent.total_observations if memory else 0,used)

def training_tasks():
    return [
      Task('train_h1',(1,4),(8,4),frozenset({(4,4),(4,5)})),
      Task('train_h2',(1,6),(8,6),frozenset({(5,6),(5,5)})),
      Task('train_v1',(4,1),(4,8),frozenset({(4,5),(5,5)})),
      Task('train_v2',(6,1),(6,8),frozenset({(6,4),(5,4)})),]

def evaluation_tasks():
    return [
      Task('eval_h1',(2,3),(9,3),frozenset({(6,3),(6,2)})),
      Task('eval_h2',(2,7),(9,7),frozenset({(7,7),(7,8)})),
      Task('eval_rot',(3,2),(3,9),frozenset({(3,6),(2,6)})),
      Task('eval_reflect',(8,2),(8,9),frozenset({(8,6),(9,6)})),
      Task('eval_mismatch',(1,5),(9,5),frozenset({(5,5),(5,7)})),]

def run_mode(mode):
    mem=AgentMemory()
    if mode!=Mode.NO_MEMORY:
        for t in training_tasks(): execute(t,mode,mem)
    if mode==Mode.LEARNED_ABLATION: mem=AgentMemory()
    return [execute(t,mode,mem) for t in evaluation_tasks()]

def summarize(rs):
    n=len(rs); solved=sum(r.solved for r in rs)
    print(f'  evaluation solved: {solved}/{n} ({100*solved/n:.1f}%)')
    print(f'  avg actions: {sum(r.actions for r in rs)/n:.2f}')
    print(f'  avg failures: {sum(r.failures for r in rs)/n:.2f}')
    print(f'  avg prior knowledge cells: {sum(r.prior_cells for r in rs)/n:.2f}')
    print(f'  influence events: {sum(r.influence_events for r in rs)}')
    print(f'  route changes: {sum(r.route_changes for r in rs)}')
    print(f'  learned observations: {max(r.learned_observations for r in rs)}')
    for r in rs: print(f'  {r.task}: solved={r.solved} actions={r.actions} failures={r.failures} prior={r.prior_cells} influence={r.influence_events} changes={r.route_changes}')

def main():
    print('DIA V16 - LATENT RELATIONSHIP TRANSFER')
    print('No family labels are supplied to the learner.')
    print('Training/evaluation use different obstacle coordinates.')
    print('Rotation/reflection transfer and negative-transfer controls are included.')
    print('A pattern requires repeated independent observations before use.\n')
    allr={}
    for mode in Mode:
        print(f'MODE: {mode.value}'); rs=run_mode(mode); allr[mode]=rs; summarize(rs); print()
    base={r.task:r for r in allr[Mode.NO_MEMORY]}
    print('PAIRED EVALUATION DIFFERENCES')
    for mode in (Mode.EPISODIC_MEMORY,Mode.LEARNED_RELATION,Mode.LEARNED_ABLATION):
        da=[r.actions-base[r.task].actions for r in allr[mode]]; df=[r.failures-base[r.task].failures for r in allr[mode]]
        print(f'\n{mode.value}:'); print(f'  mean action difference: {sum(da)/len(da):+.2f}'); print(f'  mean failure difference: {sum(df)/len(df):+.2f}'); print(f'  fewer actions: {sum(x<0 for x in da)}/{len(da)}'); print(f'  fewer failures: {sum(x<0 for x in df)}/{len(df)}')
    lf=sum(r.failures for r in allr[Mode.LEARNED_RELATION]); af=sum(r.failures for r in allr[Mode.LEARNED_ABLATION])
    print('\nSCIENTIFIC CHECKS'); print(f'  learned total failures: {lf}'); print(f'  ablation total failures: {af}')
    print('  PASS: learned representation changes behavior relative to ablation.' if lf<af else '  NOTE: learned representation did not reduce failures in this run.')
    print('\nIMPORTANT')
    print('This is an engineered transfer-capability test, not evidence of autonomous discovery or broad intelligence.')
    print('Next stage can remove the canonical symmetry assumption and learn representations directly from raw trajectories.')

if __name__=='__main__': main()
