"""DIA Route Planning V4: reachability-aware evaluation."""
from collections import deque
from dataclasses import dataclass
from environment.world1.world import World1
from environment.world1.entities import Position, Obstacle

W=H=10
MAX_ACTIONS=150
DIR={"north":(0,1),"east":(1,0),"south":(0,-1),"west":(-1,0)}
ORDER=list(DIR)

@dataclass
class Scenario:
    name:str
    start:tuple
    target:tuple
    orientation:str
    obstacles:set

def inside(p): return 0<=p[0]<W and 0<=p[1]<H
def add(p,d):
    v=DIR[d]; return (p[0]+v[0],p[1]+v[1])
def between(a,b):
    delta=(b[0]-a[0],b[1]-a[1])
    for d,v in DIR.items():
        if v==delta:return d
    return None

def bfs(start,target,blocked):
    if start in blocked or target in blocked:return []
    q=deque([start]); prev={start:None}
    while q:
        cur=q.popleft()
        if cur==target:break
        for d in ORDER:
            nxt=add(cur,d)
            if inside(nxt) and nxt not in blocked and nxt not in prev:
                prev[nxt]=cur;q.append(nxt)
    if target not in prev:return []
    out=[];cur=target
    while cur is not None:out.append(cur);cur=prev[cur]
    return out[::-1]

def turn_action(cur,want):
    a=ORDER.index(cur);b=ORDER.index(want)
    cw=(b-a)%4;ccw=(a-b)%4
    return "turn_right" if cw<=ccw else "turn_left"

def configure(world,s):
    T=type(world.organism.position)
    world.organism.position=T(x=s.start[0],y=s.start[1])
    world.organism.orientation=s.orientation
    world.obstacles.clear()
    for x,y in s.obstacles:
        world.obstacles.append(Obstacle(Position(x=x,y=y)))

def classify(reached,truth,no_route,budget):
    if reached and truth:return "SUCCESS_SOLVABLE"
    if no_route and not truth:return "CORRECTLY_DETECTED_UNREACHABLE"
    if no_route and truth:return "FALSE_NO_ROUTE_MODEL_ERROR"
    if budget and truth:return "FAILURE_SOLVABLE"
    if budget and not truth:return "UNRESOLVED_UNREACHABLE"
    return "UNCLASSIFIED"

def run(s):
    world=World1();configure(world,s)
    known=set();actions=good=bad=turns=replans=0;no_route=False
    while actions<MAX_ACTIONS:
        cur=(world.organism.position.x,world.organism.position.y)
        if cur==s.target:break
        route=bfs(cur,s.target,known);replans+=1
        if len(route)<2:no_route=True;break
        want=between(cur,route[1])
        if world.organism.orientation!=want:
            world.step(turn_action(world.organism.orientation,want))
            actions+=1;turns+=1;continue
        before=(world.organism.position.x,world.organism.position.y)
        world.step("move_forward");after=(world.organism.position.x,world.organism.position.y)
        actions+=1
        if before!=after:good+=1
        else:
            bad+=1
            attempted=add(before,want)
            if inside(attempted):known.add(attempted)
    final=(world.organism.position.x,world.organism.position.y)
    truth=bool(bfs(s.start,s.target,s.obstacles))
    reached=final==s.target
    budget=actions>=MAX_ACTIONS and not reached and not no_route
    correct=known&s.obstacles
    fp=known-s.obstacles
    fn=s.obstacles-known
    return {"scenario":s.name,"start":s.start,"target":s.target,"final":final,
            "truth_reachable":truth,"reached":reached,"internal_no_route":no_route,
            "actions":actions,"successful_moves":good,"failed_moves":bad,
            "turns":turns,"replans":replans,"known_blocked":sorted(known),
            "false_positives":sorted(fp),"false_negatives":sorted(fn),
            "precision":len(correct)/len(known) if known else 1.0,
            "recall":len(correct)/len(s.obstacles) if s.obstacles else 1.0,
            "classification":classify(reached,truth,no_route,budget)}

def scenarios():
    return [
        Scenario("open_map",(1,1),(8,8),"north",set()),
        Scenario("central_wall_solvable",(1,5),(8,5),"east",{(4,5),(4,4),(4,6)}),
        Scenario("detour_required",(2,2),(7,2),"east",{(3,2),(4,2),(4,3),(5,3),(6,3),(6,2)}),
        Scenario("target_surrounded_unreachable",(1,1),(7,7),"north",{(6,7),(7,6),(8,7),(7,8)}),
        Scenario("target_is_obstacle",(1,1),(5,5),"north",{(5,5),(4,5),(5,4)}),
        Scenario("barrier_with_gap",(1,2),(8,7),"north",{(2,4),(3,4),(4,4),(5,4),(6,4),(7,4),(8,4)}),
    ]

def main():
    results=[]
    print("="*68);print("DIA ROUTE PLANNING V4");print("Reachability-Aware Evaluation");print("="*68)
    for s in scenarios():
        r=run(s);results.append(r)
        print("\n"+"="*68);print("SCENARIO:",s.name);print("="*68)
        for k,v in r.items():print(f"{k}: {v}")
    counts={}
    for r in results:counts[r["classification"]]=counts.get(r["classification"],0)+1
    solvable=[r for r in results if r["truth_reachable"]]
    impossible=[r for r in results if not r["truth_reachable"]]
    solved=sum(r["classification"]=="SUCCESS_SOLVABLE" for r in solvable)
    detected=sum(r["classification"]=="CORRECTLY_DETECTED_UNREACHABLE" for r in impossible)
    print("\n"+"#"*68);print("AGGREGATE REPORT");print("#"*68)
    for k,v in sorted(counts.items()):print(f"{k}: {v}/{len(results)}")
    print("Solvable trials:",len(solvable))
    print("Unreachable trials:",len(impossible))
    print("Reachable-task success rate:",f"{solved/len(solvable)*100 if solvable else 0:.1f}%")
    print("Unreachable detection rate:",f"{detected/len(impossible)*100 if impossible else 0:.1f}%")

if __name__=="__main__":main()
