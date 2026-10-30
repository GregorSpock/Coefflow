import json
import os
from pathlib import Path

from engine import version
from engine.workflow import Connection
from ui.canvas import MIN_ZOOM, MAX_ZOOM
from ui.node import Node


# A workflow file is a small JSON file holding the view, each node's module, position and
# settings, and the connections between nodes. Data and results are not stored: input files are
# referenced by path and results are recomputed on Run, so files stay tiny and save instantly.
# VERSION is the layout of the file. coefflow_version in the file is the version of Coefflow that
# last saved it, so opening it in an older copy can warn that it may use things that copy lacks
VERSION = 1
PATH_KEYS = ["path", "folder"]


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


def node_record(node, folder):
    return {
        "id": node.id,
        "name": node.name,
        "x": round(node.x, 1),
        "y": round(node.y, 1),
        "settings": relative_settings(node.settings, folder),
    }


def connection_record(connection):
    return [connection.source.id, connection.output_index, connection.target.id, connection.input_index]


# The nodes and connections as text, used to tell whether anything has changed since the last
# save. Nodes are sorted by ID so bringing one to the front does not count as a change
def snapshot(canvas):
    nodes = sorted((node_record(node, ".") for node in canvas.nodes), key=lambda record: record["id"])
    connections = sorted(connection_record(c) for c in canvas.connections)
    return json.dumps([nodes, connections], sort_keys=True, default=str)


# Writes the workflow to a file. It is written to a temporary file first and then swapped in,
# so a crash part way through never leaves a half-written workflow behind
def save(canvas, path):
    folder = os.path.dirname(os.path.abspath(path))
    centre_x, centre_y = canvas.to_canvas(*canvas.rect.center)
    workflow = {
        "version": VERSION,
        "coefflow_version": version.read(),
        "view": {"x": round(centre_x, 1), "y": round(centre_y, 1), "zoom": round(canvas.zoom, 4)},
        "next_id": canvas.next_id,
        "nodes": [node_record(node, folder) for node in canvas.nodes],
        "connections": [connection_record(c) for c in canvas.connections],
    }

    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(workflow, file, separators=(",", ":"))
    os.replace(temporary, path)


# Reads a workflow file onto the canvas, replacing what is there. Everything is checked before the
# canvas changes, so a broken file leaves the current workflow alone. Returns a list of problems
# worth telling the user about, such as data files that no longer exist
def load(canvas, path):
    folder = os.path.dirname(os.path.abspath(path))
    with open(path, encoding="utf-8") as file:
        try:
            workflow = json.load(file)
        except json.JSONDecodeError as error:
            raise ValueError(f"This is not a valid workflow file ({error})")

    if not isinstance(workflow, dict) or "nodes" not in workflow:
        raise ValueError("This is not a workflow file")
    if workflow.get("version", 1) > VERSION:
        raise ValueError("This workflow was saved by a newer version of Coefflow")

    problems = []
    saved_with = workflow.get("coefflow_version")
    current = version.read()
    if saved_with and current and version.newer(saved_with, current):
        problems.append(
            f"This workflow was last saved with Coefflow v{saved_with}, which is newer than this copy (v{current}). "
            "It may use modules or settings this version doesn't have, so check it before relying on it."
        )

    nodes = {}
    for record in workflow["nodes"]:
        try:
            node = Node(record["name"], float(record["x"]), float(record["y"]), int(record["id"]))
        except (KeyError, TypeError, ValueError):
            raise ValueError("A module in the file is missing its name, ID or position")
        if node.id in nodes:
            raise ValueError(f"Two modules in the file share the ID {node.id}")
        node.settings = absolute_settings(dict(record.get("settings") or {}), folder)
        nodes[node.id] = node

        if node.module is None:
            problems.append(f"{node.name} (#{node.id}) is not a core module or in a loaded add-on. Add its pack in Settings > Add-ons, then open the workflow again.")
        for key in PATH_KEYS:
            if node.settings.get(key) and not os.path.exists(node.settings[key]):
                problems.append(f"{node.name} (#{node.id}) uses {node.settings[key]}, which was not found.")

    connections = []
    for record in workflow.get("connections", []):
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
        connections.append(Connection(source, output_index, target, input_index))

    canvas.nodes = list(nodes.values())
    canvas.connections = connections
    for node in canvas.nodes:
        canvas.tidy_inputs(node)
    canvas.next_id = max([workflow.get("next_id", 1)] + [node.id + 1 for node in canvas.nodes])

    view = workflow.get("view", {})
    canvas.zoom = max(MIN_ZOOM, min(MAX_ZOOM, float(view.get("zoom", 1.0))))
    canvas.offset_x = canvas.rect.centerx - float(view.get("x", 0)) * canvas.zoom
    canvas.offset_y = canvas.rect.centery - float(view.get("y", 0)) * canvas.zoom

    if not saved_with and current:
        add_version(path, workflow, current)
    return problems


# Adds the Coefflow version to a workflow file made before files recorded it, leaving the rest as it
# was. It is written to a temporary file first and swapped in, like save. A file that can't be
# written, e.g. one that is read-only, is left alone
def add_version(path, workflow, current):
    stamped = {"version": workflow.get("version", VERSION), "coefflow_version": current}
    stamped.update({key: value for key, value in workflow.items() if key not in stamped})

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
