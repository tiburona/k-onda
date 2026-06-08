from copy import deepcopy
from collections import deque, defaultdict


def build_generations(leaf, func=None, starting_val=None):

    generations = []
    queue = deque()
    queue.append((leaf, starting_val))

    while queue:
        generation = []
        for _ in range(len(queue)):
            node, value = queue.popleft()
            new_value = func(node, value) if func else value
            generation.append((node, new_value))

            if hasattr(node, "inputs"):
                for inp in node.inputs:
                    queue.append((inp, new_value))
        generations.append(generation)
    return generations


def walk_tree(leaf, func=None, starting_val=None):

    stack = [(leaf, starting_val, None)]
    visited = set()

    while stack:
        current_node, value, last_node = stack.pop()

        if id(current_node) in visited:
            continue

        visited.add(id(current_node))

        new_value = func(current_node, last_node, value) if func else value

        yield current_node, new_value, last_node

        for inp in getattr(current_node, "inputs", []):
            stack.append((inp, deepcopy(new_value), current_node))


def list_nodes(leaf):
    return [node for node, _, _ in walk_tree(leaf)]


def build_consumers_map(leaf):
    consumers = defaultdict(list)
    for node in list_nodes(leaf):
        for inp in getattr(node, "inputs", []):
            consumers[id(inp)].append(node)
    return consumers


def new_tree(leaf, memo=None):
    memo = {} if memo is None else memo
    new_leaf = deepcopy(leaf, memo)

    if len(leaf.inputs) == 0:
        return new_leaf

    queue1 = deque([leaf])
    queue2 = deque([new_leaf])

    while queue1:
        for _ in range(len(queue1)):
            node_1 = queue1.popleft()
            node_2 = queue2.popleft()

            for i, node in enumerate(node_1.inputs):
                if not hasattr(node, "inputs"):  # node is a root
                    node_2.inputs = list(node_2.inputs)
                    node_2.inputs[i] = node
                else:
                    queue1.append(node)
                    queue2.append(node_2.inputs[i])

    return new_leaf


def rebuild_tree(leaf, rebuild_node, memo=None):
    memo = {} if memo is None else memo

    if id(leaf) in memo:
        return memo[id(leaf)]
    
    rebuilt_inputs = tuple(rebuild_tree(inp, rebuild_node, memo) for inp in leaf.inputs)

    rebuilt = leaf if leaf.is_source else leaf.rebuild_with_inputs(rebuilt_inputs)

    if rebuild_node is not None:
        rebuilt = rebuild_node(original=leaf, rebuilt=rebuilt)

    memo[id(leaf)] = rebuilt
    return rebuilt


def walk_graph(leaf, initial_state, *, step, merge_state=None, revisit_merged=True):
    """
    initial_state: an accumulator for state that can change through traversal
    step: a function of the current node, the current state, and the consumer that
    generates the next state
    merge_state: a function of the current node, the stored state for that node, and
    the current computed state that decides how to compute state when a node is revisited
    revisit_merged: a boolean to indicate whether on merge the nodes inputs are rewalked. 
    (At some point this may need to be expanded into an enumerated policy.)
    """
    states_by_node = {}

    def walk(node, state, consumer=None):
        node_id = id(node)

        if node_id in states_by_node:
            if merge_state is not None:
                state = merge_state(node, states_by_node[node_id], state)
                states_by_node[node_id] = state
            if not revisit_merged:
                return
        else:
            states_by_node[node_id] = state
    
        next_state = step(node, state, consumer)

        for input_node in getattr(node, "inputs", []):
            walk(input_node, deepcopy(next_state), consumer=node)

    walk(leaf, deepcopy(initial_state), consumer=None)

    return states_by_node


    




    

