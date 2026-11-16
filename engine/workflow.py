from engine import ports
from engine.node import Node


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


class Workflow:
    # The nodes of one workflow and the connections between them. Every change to what is
    # connected to what goes through here, so the rules (no loops, one line into each input,
    # growing inputs kept tidy) hold however the change was asked for
    def __init__(self):
        self.nodes = []
        self.connections = []
        self.next_id = 1

    # Adds a new node at a canvas position, with the next free ID, and returns it
    def add_node(self, name, x, y):
        node = Node(name, x, y, self.next_id)
        self.nodes.append(node)
        self.next_id += 1
        return node

    def node_by_id(self, node_id):
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    # Moves a node to the end of the list so it draws on top
    def bring_to_front(self, node):
        self.nodes.remove(node)
        self.nodes.append(node)

    # Removes a node along with every connection to or from it
    def remove_node(self, node):
        self.nodes.remove(node)
        targets = [c.target for c in self.connections if c.source is node]
        self.connections = [
            c for c in self.connections if c.source is not node and c.target is not node
        ]
        for target in targets:
            self.tidy_inputs(target)

    def disconnect(self, connection):
        self.connections.remove(connection)
        self.tidy_inputs(connection.target)

    # Numbers a growing node's connected inputs 1, 2, 3... in the order they were, closing any gap,
    # and leaves one empty input after them to connect the next one to. Other nodes are left alone
    def tidy_inputs(self, node):
        if not node.growing:
            return
        into = sorted((c for c in self.connections if c.target is node), key=lambda c: c.input_index)
        for index, connection in enumerate(into):
            connection.input_index = index
        node.inputs = len(into) + 1

    # Returns the connection going into one input port, if there is one
    def connection_into(self, node, input_index):
        for connection in self.connections:
            if connection.target is node and connection.input_index == input_index:
                return connection
        return None

    # Why an output can't be connected to an input, or None if it can: a node can't feed itself,
    # a connection can't close a loop, and the output must give what the input takes (see engine/ports.py)
    def refusal(self, source, output_index, target, input_index):
        if source is target:
            return "A module can't be connected to itself"
        if reaches(self.connections, target, source):
            return "That connection would make a loop"
        return ports.mismatch(source, output_index, target, input_index)

    # Connects an output to an input, replacing whatever was already plugged into that input.
    # On a growing node it goes in at that input instead, moving the ones from there down one,
    # so dragging a line to another input reorders them.
    # Returns False, changing nothing, if the connection is refused (see refusal)
    def connect(self, source, output_index, target, input_index):
        if self.refusal(source, output_index, target, input_index):
            return False

        existing = self.connection_into(target, input_index)
        if existing and target.growing:
            for connection in self.connections:
                if connection.target is target and connection.input_index >= input_index:
                    connection.input_index += 1
        elif existing:
            self.connections.remove(existing)

        self.connections.append(Connection(source, output_index, target, input_index))
        self.tidy_inputs(target)
        return True

    # Every node that can be reached by following connections downstream from node
    def downstream(self, node):
        found = []
        to_visit = [node]
        while to_visit:
            current = to_visit.pop()
            for connection in self.connections:
                if connection.source is current and connection.target not in found:
                    found.append(connection.target)
                    to_visit.append(connection.target)
        return found

    # node and every node it depends on, by following connections upstream
    def upstream(self, node):
        found = [node]
        to_visit = [node]
        while to_visit:
            current = to_visit.pop()
            for connection in self.connections:
                if connection.target is current and connection.source not in found:
                    found.append(connection.source)
                    to_visit.append(connection.source)
        return found


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
