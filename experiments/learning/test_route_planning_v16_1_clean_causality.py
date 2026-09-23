from dataclasses import dataclass
from enum import Enum
from collections import Counter,deque
from typing import Optional

W=H=10; MAX=160
Cell=tuple[int,int]; Off=tuple[int,int]
class Mode(str,Enum):
 NO='no_memory'; EP='episodic_memory'; LR='learned_relation'; AB='learned_ablation'
@dataclass(frozen=True)
class Task: name:str; start:Cell; target:Cell; obstacles:frozenset[Cell]
@dataclass
class R:
 task:str; mode:str; solved:bool; actions:int; failures:int; prior:int; patterns:int; new:int; influence:int; prior_changes:int; discovery:int; learned:int; usable:int; used:int

def inside(c): return 1<=c[0]<=W and 1<=c[1]<=H
def nbr(c): return [(c[0]+1,c[1]),(c[0]-1,c[1]),(c[0],c[1]+1),(c[0],c[1]-1)]
def bfs(s,t,blocked):
 if s in blocked or t in blocked:return None
 q=deque([s]); p={s:None}
 while q:
  c=q.popleft()
  if c==t:
   out=[]
   while c is not None: out.append(c); c=p[c]
   return out[::-1]
  for n in nbr(c):
   if inside(n) and n not in blocked and n not in p:p[n]=c;q.append(n)
 return None

def tr(o,r,f):
 x,y=o
 if f:x=-x
 for _ in range(r%4):x,y=-y,x
 return x,y
def canon(o):return min(tr(o,r,f) for f in (0,1) for r in range(4))
class Rel:
 def __init__(self):self.c=Counter();self.total=0
 def observe(self,a,b):self.c[canon((b[0]-a[0],b[1]-a[1]) )]+=1;self.total+=1
 def usable(self):return {x for x,n in self.c.items() if n>=2}
 def predict(self,pos):
  out=set()
  for o in self.usable():
   for f in (0,1):
    for r in range(4):
     x,y=tr(o,r,f); q=(pos[0]+x,pos[1]+y)
     if inside(q):out.add(q)
  return out
class Mem:
 def __init__(self):self.cells=set();self.rel=Rel()
 def learn(self,a,b):self.cells.add(b);self.rel.observe(a,b)

def train(task,mode,m):
 pos=task.start;known=set();steps=0
 while pos!=task.target and steps<MAX:
  route=bfs(pos,task.target,known)
  if not route or len(route)<2:break
  n=route[1];steps+=1
  if n in task.obstacles:
   if mode in (Mode.EP,Mode.LR):m.learn(pos,n)
   known.add(n)
  else:pos=n

def eval_task(task,mode,m):
 pos=task.start;actions=failures=inf=pc=disc=0;new=set()
 prior_cells=set(m.cells) if mode==Mode.EP and m else set()
 prior_patterns=set(m.rel.usable()) if mode==Mode.LR and m else set()
 prior_pred=m.rel.predict(pos) if mode==Mode.LR and m else set()
 prior=set(prior_cells)|set(prior_pred)
 while pos!=task.target and actions<MAX:
  base=bfs(pos,task.target,set())
  constrained=bfs(pos,task.target,prior)
  if base!=constrained:inf+=1;pc+=1
  execution=set(new)
  if mode==Mode.EP:execution|=prior_cells
  elif mode==Mode.LR:execution|=prior_pred
  route=bfs(pos,task.target,execution) or base
  if not route or len(route)<2:break
  n=route[1];actions+=1
  if n in task.obstacles:
   failures+=1
   if n not in new:new.add(n);disc+=1
   if m and mode in (Mode.EP,Mode.LR):m.learn(pos,n)
  else:pos=n
 return R(task.name,mode.value,pos==task.target,actions,failures,len(prior),len(prior_patterns),len(new),inf,pc,disc,m.rel.total if m else 0,len(m.rel.usable()) if m else 0,len(prior_patterns))

def train_tasks():return [
 Task('train_h1',(1,4),(8,4),frozenset({(4,4),(4,5)})),
 Task('train_h2',(1,6),(8,6),frozenset({(5,6),(5,5)})),
 Task('train_v1',(4,1),(4,8),frozenset({(4,5),(5,5)})),
 Task('train_v2',(6,1),(6,8),frozenset({(6,4),(5,4)}))]
def eval_tasks():return [
 Task('eval_h1',(2,3),(9,3),frozenset({(6,3),(6,2)})),
 Task('eval_h2',(2,7),(9,7),frozenset({(7,7),(7,8)})),
 Task('eval_rot',(3,2),(3,9),frozenset({(3,6),(2,6)})),
 Task('eval_reflect',(8,2),(8,9),frozenset({(8,6),(9,6)})),
 Task('eval_mismatch',(1,5),(9,5),frozenset({(5,5),(5,7)}))]

def run(mode):
 m=None if mode==Mode.NO else Mem()
 if m:
  for t in train_tasks():train(t,mode,m)
  if mode==Mode.AB:m=Mem()
 return [eval_task(t,mode,m) for t in eval_tasks()]

def summary(rs):
 n=len(rs);sv=sum(r.solved for r in rs)
 print(f'  evaluation solved: {sv}/{n} ({100*sv/n:.1f}%)')
 print(f'  avg actions: {sum(r.actions for r in rs)/n:.2f}')
 print(f'  avg failures: {sum(r.failures for r in rs)/n:.2f}')
 print(f'  avg PRIOR knowledge cells: {sum(r.prior for r in rs)/n:.2f}')
 print(f'  avg PRIOR learned patterns: {sum(r.patterns for r in rs)/n:.2f}')
 print(f'  avg NEW current-trial observations: {sum(r.new for r in rs)/n:.2f}')
 print(f'  PRIOR-memory influence events: {sum(r.influence for r in rs)}')
 print(f'  PRIOR-memory route changes: {sum(r.prior_changes for r in rs)}')
 print(f'  replans caused by NEW discoveries: {sum(r.discovery for r in rs)}')
 print(f'  learned observations stored: {max((r.learned for r in rs),default=0)}')
 print(f'  usable learned patterns stored: {max((r.usable for r in rs),default=0)}')
 for r in rs:print(f'  {r.task}: solved={r.solved} actions={r.actions} failures={r.failures} prior={r.prior} patterns={r.patterns} new={r.new} influence={r.influence} discovery_replans={r.discovery}')

def main():
 print('DIA V16.1 - CLEAN CAUSALITY CONTROL')
 print('NO_MEMORY performs no learning and receives no memory.')
 print('PRIOR knowledge is snapshotted before each evaluation trial.')
 print('Current-trial discoveries cannot count as prior-memory influence.')
 print('Ordinary replanning and prior-memory influence are reported separately.\n')
 allr={}
 for mode in Mode:
  print('MODE:',mode.value);rs=run(mode);allr[mode]=rs;summary(rs);print()
 base={r.task:r for r in allr[Mode.NO]}
 print('PAIRED EVALUATION DIFFERENCES')
 for mode in (Mode.EP,Mode.LR,Mode.AB):
  rs=allr[mode];ad=[r.actions-base[r.task].actions for r in rs];fd=[r.failures-base[r.task].failures for r in rs]
  print('\n'+mode.value+':')
  print(f'  mean action difference: {sum(ad)/len(ad):+.2f}')
  print(f'  mean failure difference: {sum(fd)/len(fd):+.2f}')
  print(f'  fewer actions: {sum(x<0 for x in ad)}/{len(ad)}')
  print(f'  fewer failures: {sum(x<0 for x in fd)}/{len(fd)}')
 lr=allr[Mode.LR];ab=allr[Mode.AB]
 lf=sum(r.failures for r in lr);af=sum(r.failures for r in ab);li=sum(r.influence for r in lr);ai=sum(r.influence for r in ab)
 print('\nCAUSAL VALIDATION')
 print('  learned failures:',lf);print('  ablation failures:',af);print('  learned prior-memory influence:',li);print('  ablation prior-memory influence:',ai)
 print('  PASS: prior learned knowledge causally influenced the learned-relation controller.' if li>0 and ai==0 else '  NOTE: causal influence was not demonstrated.')
 print('  PASS: learned relation reduced failures relative to ablation.' if lf<af else '  NOTE: learned relation did not reduce failures relative to ablation.')
 print('\nSCIENTIFIC BOUNDARY')
 print('V16.1 validates measurement and causal-control plumbing.')
 print('The representation remains engineered.')
 print('A successful result does NOT establish autonomous discovery or broad intelligence.')
 print('Only after these controls pass should V17 remove the engineered representation and test representation discovery.')
if __name__=='__main__':main()
