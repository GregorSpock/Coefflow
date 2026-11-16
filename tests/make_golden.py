# Rewrites the golden summaries. Run it only when a result is meant to have changed, and check the
# difference before keeping it:
#   python -m tests.make_golden                     the add-on cases
#   python -m tests.make_golden 09_reshape_and_diagnostics   a sample workflow, by its file name
import sys
import tempfile

from engine import addons, runner, workspace
from tests.addon_cases import CASES, run_case
from tests.support import ROOT, summarise_outputs, write_golden


def write_addons():
    summary = {}
    for name in CASES:
        try:
            first = summarise_outputs(run_case(name))
            again = summarise_outputs(run_case(name))
            if first != again:
                print(f"{name}: gives different results each run")
            summary[name] = first
        except Exception as error:
            print(f"{name}: FAILED {type(error).__name__}: {error}")
    write_golden("addons", summary)
    print(f"{len(summary)} of {len(CASES)} cases written")


# Runs a sample workflow with its Output Data pointed at a scratch folder, as the tests do
def write_workflow(stem):
    addons.load_all()
    path = ROOT / "sample_workflows" / f"{stem}.confw"
    workflow, view, problems = workspace.load(path, stamp=False)
    with tempfile.TemporaryDirectory() as folder:
        for node in workflow.nodes:
            if "folder" in node.settings:
                node.settings["folder"] = folder
        results, errors = runner.run_workflow(workflow)
    if errors:
        print(f"{stem}: errors {errors}")
        return
    nodes = {
        str(node.id): {"error": node.error, "name": node.name, "outputs": summarise_outputs(results.get(node.id, []))}
        for node in workflow.nodes
    }
    write_golden(stem, {"problems": problems, "nodes": nodes})
    print(f"{stem} written")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for stem in sys.argv[1:]:
            write_workflow(stem)
    else:
        write_addons()
