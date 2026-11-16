import copy
import json
import os
from pathlib import Path

from engine import file_formats, ports, version
from engine.node import Node
from engine.workflow import Connection, Workflow


# A workflow file is a small JSON file holding the view, each node's module, position and
# settings, and the connections between nodes. Data and results are not stored: input files are
# referenced by path and results are recomputed on Run, so files stay tiny and save instantly.
# VERSION is the layout of the file. coefflow_version in the file is the version of Coefflow that
# last saved it, so opening it in an older copy can warn that it may use things that copy lacks.
# The view is where the canvas was looking: {"x", "y"} is the canvas point at the middle of the
# window and "zoom" how far in it was
VERSION = 1
PATH_KEYS = ["path", "folder", "tex_path"]
DEFAULT_VIEW = {"x": 0.0, "y": 0.0, "zoom": 1.0}


# Copies a node's settings with its file and folder paths made relative to the workflow's folder,
# so a project folder holding the workflow and its data can be moved or shared without breaking.
# Paths on another drive cannot be made relative, so they stay absolute
def relative_settings(settings, folder):
    saved = dict(settings)
    for key in PATH_KEYS:
        if saved.get(key):
            try:
                saved[key] = Path(os.path.relpath(saved[key], folder)).as_posix()
            except ValueError:
                pass
    return saved


# Turns relative file and folder paths back into full paths. Absolute paths are left as they are
def absolute_settings(settings, folder):
    for key in PATH_KEYS:
        if settings.get(key):
            settings[key] = os.path.normpath(os.path.join(folder, settings[key]))
    return settings


# Settings without any password: a database server's address (see modules/database) is kept
# without the password it may have been typed with, so one can never reach a workflow file, a
# temp file, the undo history or the clipboard. The password itself is remembered for the session
def without_passwords(settings):
    if isinstance(settings.get("url"), str):
        settings["url"] = file_formats.remember_password(settings["url"])
    return settings


def node_record(node, folder):
    return {
        "id": node.id,
        "name": node.name,
        "x": round(node.x, 1),
        "y": round(node.y, 1),
        "settings": without_passwords(relative_settings(node.settings, folder)),
    }


def connection_record(connection):
    return [connection.source.id, connection.output_index, connection.target.id, connection.input_index]


# A node as it is kept in memory for undo, redo, copy and paste: like node_record, but with
# its paths left as they are and its settings copied, so later changes don't reach the record
def memory_record(node):
    settings = without_passwords(copy.deepcopy(node.settings))
    return {"id": node.id, "name": node.name, "x": node.x, "y": node.y, "settings": settings}


# The nodes and connections as text, used to tell whether anything has changed since the last
# save. Nodes are sorted by ID so bringing one to the front does not count as a change
def snapshot(workflow):
    nodes = sorted((node_record(node, ".") for node in workflow.nodes), key=lambda record: record["id"])
    connections = sorted(connection_record(c) for c in workflow.connections)
    return json.dumps([nodes, connections], sort_keys=True, default=str)


# Everything needed to put a workflow back exactly as it is, kept in memory rather than in a
# file: undo and redo step between these, and copy and paste use the same records for nodes
def state(workflow):
    return {
        "next_id": workflow.next_id,
        "nodes": [memory_record(node) for node in workflow.nodes],
        "connections": [connection_record(c) for c in workflow.connections],
    }


# Puts a workflow back to a state made by state(). A node that is in both, with the same
# settings, is kept as it is along with its results; one whose settings differ keeps its place
# but forgets its results, as they no longer match
def restore(workflow, saved):
    current = {node.id: node for node in workflow.nodes}
    nodes = []
    for record in saved["nodes"]:
        node = current.get(record["id"])
        if node is None or node.name != record["name"]:
            node = Node(record["name"], record["x"], record["y"], record["id"])
        node.x, node.y = record["x"], record["y"]
        if node.settings != record["settings"]:
            node.settings = copy.deepcopy(record["settings"])
            node.clear_results()
        nodes.append(node)

    by_id = {node.id: node for node in nodes}
    workflow.nodes = nodes
    workflow.connections = [
        Connection(by_id[source], output_index, by_id[target], input_index)
        for source, output_index, target, input_index in saved["connections"]
    ]
    workflow.next_id = saved["next_id"]
    for node in workflow.nodes:
        workflow.tidy_inputs(node)


# Writes the workflow to a file. It is written to a temporary file first and then swapped in,
# so a crash part way through never leaves a half-written workflow behind
def save(workflow, path, view=None):
    folder = os.path.dirname(os.path.abspath(path))
    view = view or DEFAULT_VIEW
    contents = {
        "version": VERSION,
        "coefflow_version": version.read(),
        "view": {"x": round(view["x"], 1), "y": round(view["y"], 1), "zoom": round(view["zoom"], 4)},
        "next_id": workflow.next_id,
        "nodes": [node_record(node, folder) for node in workflow.nodes],
        "connections": [connection_record(c) for c in workflow.connections],
    }

    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(contents, file, separators=(",", ":"))
    os.replace(temporary, path)


# Reads a workflow file. Returns the workflow, its view and a list of problems worth telling the
# user about, such as data files that no longer exist. A file that can't be used raises
# ValueError, so a broken file never replaces a workflow that is open.
# stamp adds the Coefflow version to a file saved before files recorded it
def load(path, stamp=True):
    folder = os.path.dirname(os.path.abspath(path))
    with open(path, encoding="utf-8") as file:
        try:
            contents = json.load(file)
        except json.JSONDecodeError as error:
            raise ValueError(f"This is not a valid workflow file ({error})")

    if not isinstance(contents, dict) or "nodes" not in contents:
        raise ValueError("This is not a workflow file")
    if contents.get("version", 1) > VERSION:
        raise ValueError("This workflow was saved by a newer version of Coefflow")

    problems = []
    saved_with = contents.get("coefflow_version")
    current = version.read()
    if saved_with and current and version.newer(saved_with, current):
        problems.append(
            f"This workflow was last saved with Coefflow v{saved_with}, which is newer than this copy (v{current}). "
            "It may use modules or settings this version doesn't have, so check it before relying on it."
        )

    nodes = {}
    for record in contents["nodes"]:
        try:
            node = Node(record["name"], float(record["x"]), float(record["y"]), int(record["id"]))
        except (KeyError, TypeError, ValueError):
            raise ValueError("A module in the file is missing its name, ID or position")
        if node.id in nodes:
            raise ValueError(f"Two modules in the file share the ID {node.id}")
        node.settings = without_passwords(absolute_settings(dict(record.get("settings") or {}), folder))
        nodes[node.id] = node

        if node.module is None:
            problems.append(f"{node.name} (#{node.id}) is not a core module or in a loaded add-on. Add its pack in Settings > Add-ons, then open the workflow again.")
        for key in PATH_KEYS:
            if node.settings.get(key) and not os.path.exists(node.settings[key]):
                problems.append(f"{node.name} (#{node.id}) uses {node.settings[key]}, which was not found.")

    connections = []
    for record in contents.get("connections", []):
        try:
            source_id, output_index, target_id, input_index = (int(value) for value in record)
            source, target = nodes[source_id], nodes[target_id]
        except (KeyError, TypeError, ValueError):
            problems.append("Skipped a connection to a module that is not in the file.")
            continue
        taken = any(c.target is target and c.input_index == input_index for c in connections)
        fits = 0 <= input_index and (target.growing or input_index < target.inputs)
        if not (0 <= output_index < source.outputs and fits) or taken:
            problems.append(f"Skipped a connection from {source.name} to {target.name} that no longer fits their ports.")
            continue
        mismatch = ports.mismatch(source, output_index, target, input_index)
        if mismatch:
            problems.append(f"Skipped a connection: {mismatch}.")
            continue
        connections.append(Connection(source, output_index, target, input_index))

    workflow = Workflow()
    workflow.nodes = list(nodes.values())
    workflow.connections = connections
    for node in workflow.nodes:
        workflow.tidy_inputs(node)
    workflow.next_id = max([contents.get("next_id", 1)] + [node.id + 1 for node in workflow.nodes])

    saved_view = contents.get("view") or {}
    view = {}
    for key, default in DEFAULT_VIEW.items():
        try:
            view[key] = float(saved_view.get(key, default))
        except (TypeError, ValueError):
            view[key] = default

    if stamp and not saved_with and current:
        add_version(path, contents, current)
    return workflow, view, problems


# Adds the Coefflow version to a workflow file made before files recorded it, leaving the rest as it
# was. It is written to a temporary file first and swapped in, like save. A file that can't be
# written, e.g. one that is read-only, is left alone
def add_version(path, contents, current):
    stamped = {"version": contents.get("version", VERSION), "coefflow_version": current}
    stamped.update({key: value for key, value in contents.items() if key not in stamped})

    temporary = f"{path}.tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as file:
            json.dump(stamped, file, separators=(",", ":"))
        os.replace(temporary, path)
    except OSError:
        try:
            os.remove(temporary)
        except OSError:
            pass
