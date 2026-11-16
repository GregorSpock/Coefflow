import copy
import hashlib
import json
import os

from engine import ports, significance
from engine.workflow import gather_inputs, run_order


# Runs a workflow, working out only what has changed since the last run.
#
# Each node is run on a fingerprint: its module, its settings, the data file it reads, and the
# fingerprints of the nodes feeding it. When a node works, the fingerprint it ran on is kept in
# node.ran_as. Next time, a node whose fingerprint is the same as its ran_as would give the same
# results again, so it is skipped and its last outputs are passed on. Change one setting and only
# that node and those downstream of it run again.
#
# A run is in three steps, so the window can stay in use while a long one is under way:
#   run = Run(workflow)          plans the run: which nodes, in what order
#   run.execute()                does the work. It only reads the nodes, so this step can be on
#                                another thread; on_status(node, status) is called as each node
#                                starts and finishes, and cancel() stops it before the next node
#   run.apply()                  writes the results onto the nodes
# run_workflow does all three at once for anything that doesn't need to stay responsive.
#
# A module can set ALWAYS_RUN = True if its results can't be kept, or make it a
# function(settings) returning True when they can't. Output Data sets it so the file is written
# again every run, and a random sample with no seed so each run draws a new one. Whatever such a
# node feeds is worked out again too, as what it was given may have changed


# The size and time of the data file a node reads, so editing the file counts as a change
def file_stamp(settings):
    path = settings.get("path")
    try:
        details = os.stat(path) if path else None
    except (OSError, TypeError):
        return None
    return [details.st_mtime_ns, details.st_size] if details else None


# True if the node's module says its results can't be kept from one run to the next
def always_runs(node):
    always = getattr(node.module, "ALWAYS_RUN", False)
    return bool(always(node.settings)) if callable(always) else bool(always)


# What a node would be run on now, given the fingerprint of each node feeding it.
# sources is a list with (fingerprint, output number) for each input port, or None if empty
def fingerprint(node, sources):
    contents = [
        node.name,
        node.settings,
        file_stamp(node.settings),
        sources,
        list(significance.levels),
    ]
    text = json.dumps(contents, sort_keys=True, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# Notes on each fitted model a node gives which module made it and with what settings, so a
# module such as Bootstrap can fit the same model again on other rows
def mark_source(node, outputs):
    for value in outputs:
        if ports.is_model(value):
            value.attrs["source"] = {"module": node.name, "settings": copy.deepcopy(node.settings)}


# The fingerprint of every node in the workflow as it stands, by node ID, or None for every
# node if the workflow has a loop in it
def fingerprints(workflow):
    found = {}
    try:
        order = run_order(workflow.nodes, workflow.connections)
    except ValueError:
        return {node.id: None for node in workflow.nodes}

    for node in order:
        sources = [None] * node.inputs
        for connection in workflow.connections:
            if connection.target is node and connection.input_index < node.inputs:
                sources[connection.input_index] = (found.get(connection.source.id), connection.output_index)
        found[node.id] = fingerprint(node, sources)
    return found


# Marks the nodes whose results no longer match the workflow as not run: those whose settings,
# data file or inputs have changed since they ran. Returns True if any node was marked
def mark_stale(workflow):
    current = fingerprints(workflow)
    changed = False
    for node in workflow.nodes:
        if node.status in ("done", "error") and node.ran_as is not None and current.get(node.id) != node.ran_as:
            node.status = None
            changed = True
    return changed


class Outcome:
    # What happened to one node in a run: status is "done", "error" or "cancelled", and reused
    # is True if its last results were kept rather than worked out again. renewed is True if its
    # results may differ from last time though it ran on the same fingerprint, because it always
    # runs or is fed by a node that does
    def __init__(self, node):
        self.node = node
        self.status = None
        self.inputs = []
        self.outputs = []
        self.error = None
        self.ran_as = None
        self.reused = False
        self.renewed = False


class Run:
    # One run of a workflow. only limits it to those nodes and the nodes they depend on, as for
    # "Run to here". force works every node out again, whether or not anything has changed
    def __init__(self, workflow, only=None, force=False):
        self.connections = list(workflow.connections)
        nodes = list(workflow.nodes)
        if only is not None:
            needed = []
            for node in only:
                needed += [n for n in workflow.upstream(node) if n not in needed]
            nodes = [node for node in nodes if node in needed]
        self.connections = [c for c in self.connections if c.source in nodes and c.target in nodes]
        self.order = run_order(nodes, self.connections)
        self.force = force
        self.outcomes = {node.id: Outcome(node) for node in self.order}
        self.stop = False

    # Asks the run to stop. The node being worked on finishes, and the rest are left as they were
    def cancel(self):
        self.stop = True

    # The fingerprint a node runs on, from those of the nodes feeding it in this run
    def fingerprint_of(self, node):
        sources = [None] * node.inputs
        for connection in self.connections:
            if connection.target is node and connection.input_index < node.inputs:
                source = self.outcomes.get(connection.source.id)
                sources[connection.input_index] = (source.ran_as if source else None, connection.output_index)
        return fingerprint(node, sources)

    # True if the node's last results can be passed on as they are
    def can_reuse(self, node, ran_as):
        if self.force or always_runs(node):
            return False
        for connection in self.connections:
            source = self.outcomes[connection.source.id]
            if connection.target is node and (source.status != "done" or source.renewed):
                return False
        return node.status == "done" and node.ran_as == ran_as and len(node.last_outputs) >= node.outputs

    # Works through the nodes in order. Returns the outcomes by node ID
    def execute(self, on_status=None):
        notify = on_status or (lambda node, status: None)
        results = {}

        for node in self.order:
            outcome = self.outcomes[node.id]
            if self.stop:
                outcome.status = "cancelled"
                continue

            outcome.ran_as = self.fingerprint_of(node)
            if self.can_reuse(node, outcome.ran_as):
                outcome.status = "done"
                outcome.reused = True
                outcome.inputs = node.last_inputs
                outcome.outputs = node.last_outputs
                results[node.id] = outcome.outputs
                continue

            notify(node, "running")
            outcome.renewed = always_runs(node) or any(
                self.outcomes[c.source.id].renewed for c in self.connections if c.target is node
            )
            self.run_node(node, outcome, results)
            if outcome.status == "done":
                results[node.id] = outcome.outputs
                # A module may fill in settings as it runs, which is not a change to run again for
                outcome.ran_as = self.fingerprint_of(node)
            notify(node, outcome.status)
        return self.outcomes

    def run_node(self, node, outcome, results):
        outcome.status = "error"
        if node.module is None or not hasattr(node.module, "run"):
            outcome.error = "This module has no run function yet"
            return

        # A module can set REQUIRED_INPUTS to leave its later input ports optional
        outcome.inputs = gather_inputs(node, self.connections, results)
        required = getattr(node.module, "REQUIRED_INPUTS", node.inputs)
        if any(value is None for value in outcome.inputs[:required]):
            outcome.error = "Missing input"
            return

        try:
            ports.check_inputs(node, outcome.inputs)
            outputs = node.module.run(outcome.inputs, node.settings)
            ports.check_outputs(node, outputs)
            mark_source(node, outputs)
        except Exception as error:
            outcome.error = str(error) or type(error).__name__
            return
        outcome.outputs = list(outputs)
        outcome.status = "done"

    # Writes each node's outcome onto it. A node the run was cancelled before reaching is left
    # as it was. Returns the errors by node ID
    def apply(self):
        errors = {}
        for outcome in self.outcomes.values():
            node = outcome.node
            if outcome.status in (None, "cancelled"):
                if node.status in ("queued", "running"):
                    node.status = None
                continue
            node.status = outcome.status
            node.error = outcome.error
            node.last_inputs = outcome.inputs
            node.last_outputs = outcome.outputs
            node.ran_as = outcome.ran_as if outcome.status == "done" else None
            if outcome.error:
                errors[node.id] = outcome.error
        return errors

    # How many nodes were worked out, and how many had their last results kept
    def counts(self):
        ran = sum(1 for o in self.outcomes.values() if o.status in ("done", "error") and not o.reused)
        kept = sum(1 for o in self.outcomes.values() if o.reused)
        return ran, kept


# Runs a workflow from start to finish and writes the results onto its nodes.
# Returns the outputs and the error messages, both by node ID
def run_workflow(workflow, only=None, force=False, on_status=None):
    run = Run(workflow, only, force)
    run.execute(on_status)
    errors = run.apply()
    results = {o.node.id: o.outputs for o in run.outcomes.values() if o.status == "done"}
    return results, errors
