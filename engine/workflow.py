class Connection:
    # A line from one node's output port to another node's input port
    def __init__(self, source, output_index, target, input_index):
        self.source = source
        self.output_index = output_index
        self.target = target
        self.input_index = input_index


# True if you can follow connections downstream from start and arrive at goal
def reaches(connections, start, goal):
    to_visit = [start]
    seen = []

    while to_visit:
        node = to_visit.pop()
        if node is goal:
            return True
        if node in seen:
            continue
        seen.append(node)

        for connection in connections:
            if connection.source is node:
                to_visit.append(connection.target)

    return False


# Orders the nodes so every node comes after the nodes that feed into it
def run_order(nodes, connections):
    order = []
    remaining = list(nodes)

    while remaining:
        ready = []
        for node in remaining:
            sources = [c.source for c in connections if c.target is node]
            if all(source in order for source in sources):
                ready.append(node)

        if not ready:
            raise ValueError("The workflow contains a loop")

        for node in ready:
            order.append(node)
            remaining.remove(node)

    return order


# Collects the dataframes arriving at each of a node's input ports
def gather_inputs(node, connections, results):
    inputs = [None] * node.inputs

    for connection in connections:
        if connection.target is node and connection.source.id in results:
            outputs = results[connection.source.id]
            if connection.output_index < len(outputs):
                inputs[connection.input_index] = outputs[connection.output_index]

    return inputs


# Runs every node in order, passing each one's output dataframes downstream.
# Returns the outputs and any error messages, both keyed by node ID
def run(nodes, connections):
    results = {}
    errors = {}

    for node in run_order(nodes, connections):
        node.last_outputs = []
        if node.module is None or not hasattr(node.module, "run"):
            errors[node.id] = "This module has no run function yet"
            continue

        # A module can set REQUIRED_INPUTS to leave its later input ports optional
        inputs = gather_inputs(node, connections, results)
        node.last_inputs = inputs
        required = getattr(node.module, "REQUIRED_INPUTS", node.inputs)
        if any(dataframe is None for dataframe in inputs[:required]):
            errors[node.id] = "Missing input"
            continue

        try:
            results[node.id] = node.module.run(inputs, node.settings)
            node.last_outputs = results[node.id]
        except Exception as error:
            errors[node.id] = str(error)

    return results, errors
