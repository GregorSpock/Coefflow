import argparse
import json
import re
import sys
import time
from pathlib import Path

from engine import addons, catalogue, file_formats, runner, version, workspace


# Coefflow without the window: run a saved workflow from a terminal, a script or a scheduler.
#
#   python main.py run analysis.confw                    run it, saving whatever its Output Data saves
#   python main.py run analysis.confw --export results   also save every module's outputs to results/
#   python main.py run analysis.confw --set 3.robust=true --set 1.path=data2024.csv
#   python main.py check analysis.confw                  open it and list any problems, without running
#   python main.py modules                               list every module, core and add-on
#
# A run prints a line per module as it finishes and ends with a summary. The exit code is 0 when
# every module worked, 1 when any failed and 2 when the workflow couldn't be opened, so a script
# or CI job can tell. Nothing here imports Qt, so it works on a server with no screen

OK, FAILED, UNUSABLE = 0, 1, 2


def parser():
    main = argparse.ArgumentParser(
        prog="coefflow",
        description="Run Coefflow workflows without the window.",
        epilog="Start the app with no arguments: python main.py",
    )
    main.add_argument("--version", action="version", version=f"Coefflow {version.read() or ''}".strip())
    commands = main.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run a workflow file")
    run.add_argument("workflow", help="the .confw file to run")
    run.add_argument("--export", metavar="FOLDER", help="save every module's outputs to this folder")
    run.add_argument("--format", default="csv", choices=list(file_formats.WRITE_FORMATS), help="file type for --export (default csv)")
    run.add_argument("--set", action="append", default=[], metavar="ID.SETTING=VALUE",
                     help="change a module's setting for this run only, e.g. 3.robust=true. VALUE is read as JSON if it can be")
    run.add_argument("--only", action="append", type=int, default=[], metavar="ID",
                     help="run only this module and those it needs (can be given more than once)")
    run.add_argument("--show", action="append", type=int, default=[], metavar="ID",
                     help="print this module's outputs when the run ends (can be given more than once)")
    run.add_argument("--quiet", action="store_true", help="only print errors and the summary")

    check = commands.add_parser("check", help="open a workflow and list any problems, without running it")
    check.add_argument("workflow")

    commands.add_parser("modules", help="list every module, core and from loaded add-ons")
    return main


def say(text, quiet=False):
    if not quiet:
        print(text, flush=True)


# Opens a workflow, printing the problems found. Returns the workflow, or None if it can't be used
def open_workflow(path):
    if not Path(path).exists():
        print(f"No such workflow: {path}", file=sys.stderr)
        return None
    try:
        workflow, view, problems = workspace.load(path, stamp=False)
    except (OSError, ValueError) as error:
        print(f"Could not open {path}: {error}", file=sys.stderr)
        return None
    for problem in problems:
        print(f"Warning: {problem}", file=sys.stderr)
    return workflow


# A setting's value from the command line: JSON where it reads as JSON (true, 3, [1, 2], "x"),
# else the text as it is
def setting_value(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


# Applies each --set ID.SETTING=VALUE to the workflow's modules. Raises ValueError for one that
# doesn't name a module in the workflow
def apply_settings(workflow, changes, folder):
    for change in changes:
        match = re.fullmatch(r"(\d+)\.([^=]+)=(.*)", change, re.DOTALL)
        if not match:
            raise ValueError(f"--set {change} should look like ID.SETTING=VALUE, e.g. 3.robust=true")
        node = workflow.node_by_id(int(match.group(1)))
        if node is None:
            raise ValueError(f"--set {change}: the workflow has no module #{match.group(1)}")
        key, value = match.group(2).strip(), setting_value(match.group(3))
        if key in workspace.PATH_KEYS and isinstance(value, str):
            value = str((Path.cwd() / value).resolve()) if not Path(value).is_absolute() else value
        node.settings[key] = value


def safe_name(text):
    return re.sub(r"[^\w.-]+", "_", str(text)).strip("_") or "module"


# Saves every table each module gave to the folder, named like 3_OLS_out1.csv
def export_outputs(workflow, results, folder, kind):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    ending = file_formats.WRITE_FORMATS[kind][0]
    saved = 0
    for node in workflow.nodes:
        for number, table in enumerate(results.get(node.id, []), start=1):
            if table is None:
                continue
            path = folder / f"{node.id}_{safe_name(node.name)}_out{number}{ending}"
            file_formats.write_table(table, path, kind, title=f"{node.name} Out {number}")
            saved += 1
    return saved


def show_outputs(workflow, results, ids):
    import pandas as pd
    with pd.option_context("display.width", 160, "display.max_columns", 20, "display.max_rows", 40):
        for node_id in ids:
            node = workflow.node_by_id(node_id)
            if node is None:
                print(f"\n#{node_id}: no such module")
                continue
            for number, table in enumerate(results.get(node_id, []), start=1):
                print(f"\n#{node_id} {node.name}, Out {number} ({len(table):,} rows):")
                print(table.to_string(max_rows=40) if len(table) <= 40 else table.head(40).to_string())


def run_command(arguments):
    workflow = open_workflow(arguments.workflow)
    if workflow is None:
        return UNUSABLE
    try:
        apply_settings(workflow, arguments.set, Path(arguments.workflow).parent)
        only = None
        if arguments.only:
            only = [workflow.node_by_id(node_id) for node_id in arguments.only]
            if None in only:
                raise ValueError(f"--only: the workflow has no module #{arguments.only[only.index(None)]}")
        run = runner.Run(workflow, only=only)
    except ValueError as error:
        print(error, file=sys.stderr)
        return UNUSABLE

    started = time.perf_counter()
    say(f"Running {Path(arguments.workflow).name}: {len(run.order)} modules", arguments.quiet)

    def progress(node, status):
        if status == "done":
            say(f"  ok     #{node.id} {node.name}", arguments.quiet)
        elif status == "error":
            outcome = run.outcomes[node.id]
            print(f"  FAILED #{node.id} {node.name}: {outcome.error}", file=sys.stderr, flush=True)

    run.execute(progress)
    errors = run.apply()
    results = {o.node.id: o.outputs for o in run.outcomes.values() if o.status == "done"}
    seconds = time.perf_counter() - started

    if arguments.show:
        show_outputs(workflow, results, arguments.show)
    if arguments.export:
        saved = export_outputs(workflow, results, arguments.export, arguments.format)
        say(f"Saved {saved} tables to {arguments.export}", arguments.quiet)

    worked = sum(1 for o in run.outcomes.values() if o.status == "done")
    print(f"{worked} of {len(run.order)} modules worked, {len(errors)} failed, in {seconds:.1f}s")
    return FAILED if errors else OK


def check_command(arguments):
    workflow = open_workflow(arguments.workflow)
    if workflow is None:
        return UNUSABLE
    for node in sorted(workflow.nodes, key=lambda n: n.id):
        into = [c for c in workflow.connections if c.target is node]
        sources = ", ".join(f"#{c.source.id} Out {c.output_index + 1}" for c in sorted(into, key=lambda c: c.input_index))
        print(f"#{node.id} {node.name}" + (f"  <- {sources}" if sources else ""))
    try:
        runner.Run(workflow)
    except ValueError as error:
        print(error, file=sys.stderr)
        return UNUSABLE
    missing = [node for node in workflow.nodes if node.module is None]
    return UNUSABLE if missing else OK


def modules_command(arguments):
    for tab, names in catalogue.all_tabs().items():
        print(f"{tab}: {', '.join(names)}")
    return OK


def main(argv=None):
    arguments = parser().parse_args(argv)
    for problem in addons.load_all():
        print(f"Warning: {problem}", file=sys.stderr)
    commands = {"run": run_command, "check": check_command, "modules": modules_command}
    return commands[arguments.command](arguments)


if __name__ == "__main__":
    sys.exit(main())
