"""DIA V7 - corrected action execution and leakage-free paired evaluation."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from enum import Enum
import random, statistics
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

from environment.world1.world import World1
from environment.world1.entities import Obstacle, Position

Cell = Tuple[int, int]
MOVE = "move_forward"
RIGHT = "turn_right"
LEFT = "turn_left"
ORIENTATIONS = ["north", "east", "south", "west"]
DIRECTIONS = {"north": (0,1), "east": (1,0), "south": (0,-1), "west": (-1,0)}

class Mode(str, Enum):
    NO_MEMORY = "no_memory"
    EPISODIC = "episodic_memory"
    STRUCTURAL = "structural_memory"

@dataclass(frozen=True)
class Scenario:
    sid: str; phase: str; obstacles: FrozenSet[Cell]; start: Cell; target: Cell; orientation: str

@dataclass
class Result:
    sid: str; mode: str; reached: bool; reachable: bool; actions: int; successes: int; failures: int; turns: int
    shortest: Optional[int]; final: Cell; discovered: Set[Cell]; classification: str

class Memory:
    def __init__(self): self.records: Dict[Tuple, List[bool]] = {}
    def key(self, start: Cell, target: Cell, orientation: str) -> Tuple:
        dx = (target[0] > start[0]) - (target[0] < start[0])
        dy = (target[1] > start[1]) - (target[1] < start[1])
        return (dx, dy, orientation, start[0] in (0,9), start[1] in (0,9), target[0] in (0,9), target[1] in (0,9))
    def confidence(self, key: Tuple) -> float:
        values = self.records.get(key, [])
        return sum(values) / len(values) if values else 0.0
    def learn(self, key: Tuple, reached: bool): self.records.setdefault(key, []).append(reached)

def inside(c: Cell) -> bool: return 0 <= c[0] < 10 and 0 <= c[1] < 10
def neigh(c: Cell):
    x,y=c
    yield (x+1,y); yield (x-1,y); yield (x,y+1); yield (x,y-1)

def bfs(start: Cell, target: Cell, blocked: Set[Cell]) -> Optional[List[Cell]]:
    if start in blocked or target in blocked: return None
    q=deque([start]); prev={start:None}
    while q:
        cur=q.popleft()
        if cur==target:
            out=[]; n=cur
            while n is not None: out.append(n); n=prev[n]
            return out[::-1]
        for nxt in neigh(cur):
            if inside(nxt) and nxt not in blocked and nxt not in prev:
                prev[nxt]=cur; q.append(nxt)
    return None

def configure(w: World1, s: Scenario):
    w.obstacles.clear()
    for x,y in sorted(s.obstacles): w.obstacles.append(Obstacle(Position(x=x,y=y)))
    pt=type(w.organism.position); w.organism.position=pt(x=s.start[0], y=s.start[1])
    w.organism.orientation=s.orientation; w.organism.energy=100

def direction(a: Cell,b: Cell)->str:
    dx,dy=b[0]-a[0],b[1]-a[1]
    return {(1,0):"east",(-1,0):"west",(0,1):"north",(0,-1):"south"}[(dx,dy)]

def turns(cur: str, desired: str)->List[str]:
    if cur==desired:return []
    ci=ORIENTATIONS.index(cur); di=ORIENTATIONS.index(desired)
    r=(di-ci)%4; l=(ci-di)%4
    return [RIGHT]*r if r<=l else [LEFT]*l

def cell_of(w: World1)->Cell:
    p=w.organism.position; return (p.x,p.y)

def generate(seed=20260921, n=12):
    rng=random.Random(seed); out=[]
    for phase,density in (("training",.13),("transfer",.20)):
        for i in range(1,n+1):
            start=(rng.randrange(10),rng.randrange(10)); target=(rng.randrange(10),rng.randrange(10))
            while target==start: target=(rng.randrange(10),rng.randrange(10))
            obs={ (x,y) for x in range(10) for y in range(10) if (x,y) not in (start,target) and rng.random()<density }
            out.append(Scenario(f"{phase}_{i:02d}",phase,frozenset(obs),start,target,rng.choice(ORIENTATIONS)))
    return out

def execute(s: Scenario, mode: Mode, memory: Optional[Memory], max_actions=180)->Result:
    w=World1(); configure(w,s); model=set(); failures=set(); actions=successes=failed=turn_count=0
    key=memory.key(s.start,s.target,s.orientation) if memory else None
    while actions<max_actions:
        pos=cell_of(w); orient=w.organism.orientation
        if pos==s.target: break
        route=bfs(pos,s.target,model)
        if route is None or len(route)<2: break
        desired=direction(route[0],route[1]); needed=turns(orient,desired)
        if needed:
            # Execute the turn in the environment, rather than changing a local variable only.
            w.step(needed[0]); actions+=1; turn_count+=1
            continue
        before=pos
        predicted=(pos[0]+DIRECTIONS[orient][0],pos[1]+DIRECTIONS[orient][1])
        w.step(MOVE); actions+=1; after=cell_of(w)
        if after!=before:
            successes+=1
        else:
            failed+=1; failures.add(predicted); model.add(predicted)
    reached=cell_of(w)==s.target
    actual=set(s.obstacles); gt=bfs(s.start,s.target,actual); reachable=gt is not None
    if reached and reachable: cls="SUCCESS_SOLVABLE"
    elif not reachable and not reached: cls="CORRECTLY_DETECTED_OR_UNRESOLVED"
    elif reachable and not reached: cls="FAILURE_SOLVABLE"
    else: cls="UNCLASSIFIED"
    if memory and key is not None: memory.learn(key,reached)
    return Result(s.sid,mode.value,reached,reachable,actions,successes,failed,turn_count,len(gt)-1 if gt else None,cell_of(w),model,cls)

def main():
    seed=20260921; scenarios=generate(seed); memories={Mode.NO_MEMORY:None,Mode.EPISODIC:Memory(),Mode.STRUCTURAL:Memory()}; allres={}
    print("DIA V7 - FIXED CONTROLLER / LEAKAGE-FREE EVALUATION")
    print(f"Random seed: {seed}\nScenarios: {len(scenarios)}")
    for s in scenarios:
        allres[s.sid]={}
        for mode in Mode:
            allres[s.sid][mode.value]=execute(s,mode,memories[mode])
    for mode in Mode:
        rs=[allres[s.sid][mode.value] for s in scenarios]; solved=sum(r.reached for r in rs)
        print(f"\nMODE: {mode.value}\n  solved: {solved}/{len(rs)} ({100*solved/len(rs):.1f}%)")
        print(f"  avg actions: {statistics.mean(r.actions for r in rs):.2f}")
        print(f"  avg failures: {statistics.mean(r.failures for r in rs):.2f}")
        print(f"  avg turns: {statistics.mean(r.turns for r in rs):.2f}")
    base=Mode.NO_MEMORY.value
    for mode in (Mode.EPISODIC.value,Mode.STRUCTURAL.value):
        ad=[]; fd=[]
        for s in scenarios:
            a=allres[s.sid][base]; b=allres[s.sid][mode]; ad.append(b.actions-a.actions); fd.append(b.failures-a.failures)
        print(f"\nPAIRED: {mode}\n  mean action difference: {statistics.mean(ad):+.3f}\n  mean failure difference: {statistics.mean(fd):+.3f}\n  fewer actions: {sum(x<0 for x in ad)}/{len(ad)}\n  fewer failures: {sum(x<0 for x in fd)}/{len(fd)}")
    print("\nDIAGNOSTIC SAMPLE")
    for s in scenarios[:3]:
        for mode in Mode:
            r=allres[s.sid][mode.value]
            print(f"{s.sid:12} {mode.value:18} final={r.final} target={s.target} actions={r.actions} fail={r.failures} class={r.classification}")
    print("\nNotes: no controller receives the obstacle set. Structural memory uses only start/target/orientation features; it is engineered, not discovered.")

if __name__=="__main__": main()
