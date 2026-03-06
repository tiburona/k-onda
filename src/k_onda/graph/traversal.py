from copy import deepcopy, copy


from collections import deque




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
            
            if hasattr(node, 'inputs'):
                for inp in node.inputs:
                    queue.append((inp, new_value))
        generations.append(generation)
    return generations


def walk_tree(leaf, func=None, starting_val=None):

    stack = [(leaf, starting_val, None)]
    visited = set()

    while stack:
        
        current_node, value, last_node = stack.pop()
        
        if current_node in visited:
            continue

        visited.add(current_node)

        new_value = func(current_node, value) if func else value
        
        yield current_node, new_value, last_node

        for inp in getattr(current_node, 'inputs', []):
            stack.append((inp, deepcopy(new_value), current_node))



def new_tree(leaf):
    new_leaf = deepcopy(leaf)

    if len(leaf.inputs) == 0:
        return new_leaf
    
    queue1 = deque([leaf])
    queue2 = deque([new_leaf])

    while queue1:
        for _ in range(len(queue1)):
            node_1 = queue1.popleft()
            node_2 = queue2.popleft()

            for i, node in enumerate(node_1.inputs):
                if not hasattr(node, 'inputs'):  # node is a root
                    node_2.inputs = list(node_2.inputs)
                    node_2.inputs[i] = node
                else:
                    queue1.append(node)
                    queue2.append(node_2.inputs[i])

    return new_leaf

            