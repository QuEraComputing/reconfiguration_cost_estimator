"""
Created on Fri Jan 17 08:22:31 2025

Move cost estimators for reconfiguration and gate move steps.

There are two main functions:
gate_forward(*) - estimate the cost of doing some gates
reconfigure_layout(*) - estimate the cost of reconfiguring the layout of the qubits

@author: Jonathan Wurtz, QuEra Computing
"""
import numpy as np
import pydantic # Used for type checking only; not strictly required

class QuerrahModel:
    """
    A static class that holds information about errors.
    """
    idle_error = 1.0 # Penalty from idle time, which accounts for the total time of the moves.
    move_error = 0.1 # Penalty from "touching" an atom
    inter_zone_time = 10.0 # Time to move to and from gate region, in units of sqrt(lattice spacing)


class Qubit(int):
    def __repr__(self):
        return f"Q[{int(self)}]"

# The decorator is a guardrail to validate type inputs. Comment out for speed.
@pydantic.validate_call(config=pydantic.ConfigDict(arbitrary_types_allowed=True)) # Make python strongly typed!!!
def gate_forward(
        state: dict[Qubit,tuple[int,int]],
        gates: set[tuple[Qubit,Qubit]]) -> float:
    """
    Estimate the cost of moving atoms to and from the storage region
    in order to implement some set of gates. Ths can probably be tightened.
    This estimator assumes an "infinite" amount of swap space for reconfiguration,
    and an "infinite" 2d grid of gate pairs.
    
    Inputs:
    state - the current state of the qubits in the storage region
    gates - the set of 2 qubit gates to implement
    
    Returns:
    cost - The cost of implementing those moves. Lower is better.
    """
    
    N = len(state)
    active_sites = _get_active_gate_sites(state, gates)
    
    if len(active_sites)==0:
        return 0.0
    
    nedges = _reconfiguation_conflicts(active_sites)
    """
    # Model where the travel time depends on the maximum span of the reconfiguration
    visited_x = [q[0][0] for q in active_sites] + [q[1][0] for q in active_sites]
    visited_y = [q[0][1] for q in active_sites] + [q[1][1] for q in active_sites]
    dx = max(visited_x) - min(visited_x)
    dy = max(visited_y) - min(visited_y)
    """
    # Model where the travel time depends on the maximum distance between any two atoms
    # This is strictly less than the above, but would correspond to a slightly smarter
    # use of the reconfiguration area.
    moved_x_ = [abs(q[0][0] - q[1][0]) for q in active_sites]
    moved_y_ = [abs(q[0][1] - q[1][1]) for q in active_sites]
    dx = max(moved_x_)
    dy = max(moved_y_)
    Tc = np.sqrt(max([dx,dy]))
    
    total_moves = np.log(nedges) + 1
    total_time = Tc * total_moves + QuerrahModel.inter_zone_time
    total_touches = len(active_sites)*total_moves
    
    
    return (
        total_time * N * QuerrahModel.idle_error + \
        total_touches * QuerrahModel.move_error
    )
    
# The decorator is a guardrail to validate type inputs. Comment out for speed.
@pydantic.validate_call(config=pydantic.ConfigDict(arbitrary_types_allowed=True)) # Make python strongly typed!!!
def reconfigure_layout(
    start_state: dict[Qubit,tuple[int,int]],
    end_state: dict[Qubit,tuple[int,int]],
)-> float:
    """
    Estimate the cost of reconfiguring atoms in the storage region.
    This estimator assumes an "infinite" amount of swap space for reconfiguration
    
    Inputs:
    start_state - the current state of the qubits in the storage region.
     Keys are qubits, and values are the position of that qubit in the storage region as (x,y)
    end_state - the desired state of the qubits in the storage region
    
    Returns:
    cost - The cost of implementing those moves. Lower is better.
    """
    
    N = len(start_state)
    active_sites = _get_active_reconfiguration_sites(start_state, end_state)
    
    if len(active_sites)==0:
        return 0.0
    
    nedges = _reconfiguation_conflicts(active_sites)
    """
    # Model where the travel time depends on the maximum span of the reconfiguration
    visited_x = [q[0][0] for q in active_sites] + [q[1][0] for q in active_sites]
    visited_y = [q[0][1] for q in active_sites] + [q[1][1] for q in active_sites]
    dx = max(visited_x) - min(visited_x)
    dy = max(visited_y) - min(visited_y)
    """
    # Model where the travel time depends on the maximum distance between any two atoms
    # This is strictly less than the above, but would correspond to a slightly smarter
    # use of the reconfiguration area.
    moved_x_ = [abs(q[0][0] - q[1][0]) for q in active_sites]
    moved_y_ = [abs(q[0][1] - q[1][1]) for q in active_sites]
    dx = max(moved_x_)
    dy = max(moved_y_)
    Tc = np.sqrt(max([dx,dy]))
    
    
    
    total_moves = np.log(nedges) + 1
    total_time = Tc * total_moves
    total_touches = len(active_sites)*total_moves
    
    
    return (
        total_time * N * QuerrahModel.idle_error + \
        total_touches * QuerrahModel.move_error
    )

def _validate_state(state:dict[Qubit,tuple[int,int]]):
    if len(set(state.values())) != len(state):
        raise ValueError("State has duplicate values",state)
    return True


def _get_active_reconfiguration_sites(
    start_state: dict[Qubit,tuple[int,int]],
    end_state: dict[Qubit,tuple[int,int]],
)-> list[tuple[ tuple[int,int], tuple[int,int]]]:
    """
    Identify the atoms that participate in a reconfiguration step.
    Returns a list, where each element is a tuple of the start and end positions each atom.
    """
    _validate_state(start_state)
    _validate_state(end_state)
    
    actives = []
    qubits = []
    for qubit in start_state.keys():
        if start_state[qubit] != end_state[qubit]:
            actives.append((start_state[qubit], end_state[qubit]))
            qubits.append(qubit)
    
    # Next, identify the atoms which would get "caught" in the crossed AODs but
    # are not actually moved.
    active_rows = set([q[0][0] for q in actives]).union(set([q[1][0] for q in actives]))
    active_cols = set([q[0][1] for q in actives]).union(set([q[1][1] for q in actives]))
    for qubit in start_state.keys():
        if qubit not in qubits:
            if start_state[qubit][0] in active_rows and start_state[qubit][1] in active_cols:
                actives.append((start_state[qubit], start_state[qubit]))
            #if end_state[qubit][0] in active_rows and end_state[qubit][1] in active_cols:
            #    actives.append((end_state[qubit], end_state[qubit]))
    
    # Remove any spurrious duplicates
    actives = list(set(actives))
    return actives

def _get_active_gate_sites(
    state: dict[Qubit,tuple[int,int]],
    gates: set[tuple[Qubit,Qubit]]
)-> list[tuple[ tuple[int,int], tuple[int,int]]]:
    """
    Extract the active sites of the gates in the current state.
    Returns a list, where each element is a tuple of the position of each atom in the gate pair
    """
    actives = []
    for gate in gates:
        actives.append((state[gate[0]], state[gate[1]]))
    return actives

def _reconfiguation_conflicts(
    active_sites: list[tuple[ tuple[int,int], tuple[int,int]]],
) -> int:
    """
    Count the number of edges in the conflict graph
    Returns:
    int - the number of edges in the conflict graph
    """
    
    n_edges = 0
    for i,_ in enumerate(active_sites):
        for j in range(i+1,len(active_sites)):
            # X one-to-many
            if (active_sites[i][0][0] == active_sites[j][0][0]) ^ (active_sites[i][1][0] == active_sites[j][1][0]):
                n_edges += 1
                # print("X one-to-many detected", active_sites[i], active_sites[j])
            # Y one-to-many
            elif (active_sites[i][0][1] == active_sites[j][0][1])  ^ (active_sites[i][1][1] == active_sites[j][1][1]):
                n_edges += 1
                # print("Y one-to-many detected", active_sites[i], active_sites[j])
            
            # X ordering
            elif  (active_sites[i][0][0] < active_sites[j][0][0]) ^ (active_sites[i][1][0] < active_sites[j][1][0]):
                n_edges += 1
                # print("X ordering detected", active_sites[i], active_sites[j])
            # Y ordering
            elif  (active_sites[i][0][1] < active_sites[j][0][1]) ^ (active_sites[i][1][1] < active_sites[j][1][1]):
                n_edges += 1
                # print("Y ordering detected", active_sites[i], active_sites[j])
            
            
    return n_edges + 1
            
            
    


if __name__ == "__main__":
    """
    Tests and validations
    """
    
    print("---\n_get_active_reconfiguration_sites\n---")
    print("Test 1: empty start and end state")
    print(_get_active_reconfiguration_sites(
        start_state = {}, end_state = {}
    ))
    
    print("Test 2: single qubit moves, another does not")
    print(_get_active_reconfiguration_sites(
        start_state={Qubit(0):(0,0), Qubit(1):(0,1)},
        end_state = {Qubit(0): (0,2), Qubit(1):(0,1)}
    ))
    
    print("Test 3: Two qubits move, another gets caught in the starting cross")
    print(_get_active_reconfiguration_sites(
        start_state={Qubit(0):(0,0), Qubit(1):(1,1), Qubit(2):(0,1)},
        end_state = {Qubit(0): (5,6), Qubit(1):(7,9), Qubit(2):(0,1)}
    ))
    
    print("Test 4: Two qubits move, another gets caught in the ending cross")
    print(_get_active_reconfiguration_sites(
        start_state={Qubit(0):(0,0), Qubit(1):(1,1), Qubit(2):(7,8)},
        end_state = {Qubit(0): (5,6), Qubit(1):(7,9), Qubit(2):(7,8)}
    ))
    
    print("---\n_get_active_gate_sites\n---")
    print("Test 1: empty state")
    print(_get_active_gate_sites(
        state={}, gates=set()
    ))
    
    print("Test 2: single gate")
    print(_get_active_gate_sites(
        state={Qubit(0):(0,0), Qubit(1):(0,1), Qubit(2):(2,3)},
        gates = {(Qubit(0), Qubit(1))}
    ))
    
    print("Test 3: missing qubit")
    try:
        print(_get_active_gate_sites(
            state={Qubit(0):(0,0), Qubit(1):(0,1), Qubit(2):(2,3)},
            gates = {(Qubit(0), Qubit(3))}
        ))
    except KeyError as e:
        print("KeyError caught",e)
        
    print("---\n_reconfiguation_conflicts\n---")
    print("Test 1: no conflicts")
    print(_reconfiguation_conflicts([((0,0),(0,1)),((0,2),(0,3)),((0,4),(0,5))]))
    
    print("Test 2: one conflict")
    print(_reconfiguation_conflicts([((0,0),(0,1)),((0,1),(0,0)),((0,4),(0,5))]))
    
    print("Test 3: Many-to-one conflict")
    print(_reconfiguation_conflicts([((0,0),(0,1)),((0,1),(0,1)),((0,2),(0,1))]))
    
    print("Test 4: One-to-many conflict")
    print(_reconfiguation_conflicts([((0,2),(0,1)),((0,2),(0,2)),((0,2),(0,3))]))
    
    
    
    
    print("---\nreconfigure_layout\n---")
    print("Test 1: Nothing happens")
    print(reconfigure_layout(
        start_state={Qubit(0):(0,0)}, end_state={Qubit(0):(0,0)}
    ))
    
    print("Test 2: single qubit moves, another does not")
    print(reconfigure_layout(
        start_state={Qubit(0):(0,0), Qubit(1):(0,1)},
        end_state = {Qubit(0): (0,2), Qubit(1):(0,1)}
    ))
    
    print("Test 3: Unconflicted reconfiguration")
    print(reconfigure_layout(
        start_state={Qubit(0):(0,0), Qubit(1):(0,1)},
        end_state = {Qubit(0): (1,1), Qubit(1):(1,2)}
    ))
    
    print("Test 3: Slightly conflicted reconfiguration (move to different rows)")
    print(reconfigure_layout(
        start_state={Qubit(0):(0,0), Qubit(1):(0,1)},
        end_state = {Qubit(0): (1,1), Qubit(1):(2,2)}
    ))
    
    print("Test 4: Invert order 16")
    print(reconfigure_layout(
        start_state={Qubit(i):(i,0) for i in range(16)},
        end_state = {Qubit(i):(0,16-i) for i in range(16)}
    ))
    print("Test 4: Invert order 64")
    print(reconfigure_layout(
        start_state={Qubit(i):(i,0) for i in range(64)},
        end_state = {Qubit(i):(0,64-i) for i in range(64)}
    ))
    print("Test 5: Two qubits move, another gets caught in the starting cross")
    print(reconfigure_layout(
        start_state={Qubit(0):(0,0), Qubit(1):(1,1), Qubit(2):(0,1)},
        end_state = {Qubit(0): (2,2), Qubit(1):(3,3), Qubit(2):(0,1)}
    ))
    
    print("Test 5: Three qubits move on a grid")
    print(reconfigure_layout(
        start_state={Qubit(0):(0,0), Qubit(1):(1,1), Qubit(2):(0,1)},
        end_state = {Qubit(0): (2,2), Qubit(1):(3,3), Qubit(2):(2,3)}
    ))
    

    print("---\ngate_forward\n---")
    print("Test 1: empty state")
    # Maybe slightly wrong as we don't test for an empty gate state resulting in some
    # constant time overhead, but then again why would you ask for zero gates?
    print(gate_forward(
        state={Qubit(0):(0,0)}, gates=set()
    ))
    
    print("Test 2: Single unconflicted gate")
    print(gate_forward(
        state={Qubit(0):(0,0), Qubit(1):(0,1)}, gates={(Qubit(0), Qubit(1))}
    ))
    
    print("Test 2: Single partially unconflicted gate")
    print(gate_forward(
        state={Qubit(0):(0,0), Qubit(1):(1,1)}, gates={(Qubit(0), Qubit(1))}
    ))
    
    print("Test 2: Two conflicted gates on the same row")
    print(gate_forward(
        state={Qubit(0):(0,0), Qubit(1):(4,0), Qubit(2):(1,0), Qubit(3):(2,0)}, gates={(Qubit(0), Qubit(1)), (Qubit(2), Qubit(3))}
    ))
    
    print("Test 3: no conflicts")
    print(gate_forward(
        state={Qubit(i):(i,0) for i in range(16)}, gates={(Qubit(2*i), Qubit(2*i+1)) for i in range(8)}
    ))
    print("Test 4: complete conflicts")
    print(gate_forward(
        state={Qubit(i):(i,0) for i in range(16)}, gates={(Qubit(i), Qubit(15-i)) for i in range(8)}
    ))
    
    
`# Test random vs 2d structured configuration for a 2d layout (eg 2d Heisenberg model)
nx = 16
ny = 8
ctr = 0
state = {}

for i in range(nx):
    for j in range(ny):
        state[Qubit(ctr)] = (i,j)
        ctr += 1

state_r = {val:key for key,val in state.items()}
gates_vertical = []
gates_vertical_b = []
for i in range(nx):
    for j in range(0,ny-1,2):
        gates_vertical.append((state_r[(i,j)], state_r[(i,j+1)]))
for i in range(nx):
    for j in range(1,ny,2):
        gates_vertical_b.append((state_r[(i,j)], state_r[(i,j-1)]))
gates_horizontal = []
gates_horizontal_b = []
for i in range(0,nx-1,2):
    for j in range(ny):
        gates_horizontal.append((state_r[(i,j)], state_r[(i+1,j)]))
for i in range(1,nx,2):
    for j in range(ny):
        gates_horizontal_b.append((state_r[(i,j)], state_r[(i-1,j)]))

print("-- big test --")
print("Structured:")
print(gate_forward(state=state, gates=set(gates_horizontal)) +
        gate_forward(state=state, gates=set(gates_horizontal_b)) +
        gate_forward(state=state, gates=set(gates_vertical)) +
        gate_forward(state=state, gates=set(gates_vertical_b)))

print("Random:")
for _ in range(10):
    perm = np.random.permutation(nx*ny)
    state_random = {}
    for i in range(nx*ny):
        state_random[Qubit(i)] = state[Qubit(perm[i])]
    
    print(gate_forward(state=state_random, gates=set(gates_horizontal))+
            gate_forward(state=state_random, gates=set(gates_horizontal_b)) +
            gate_forward(state=state_random, gates=set(gates_vertical)) +
        gate_forward(state=state_random, gates=set(gates_vertical_b)))
`
    