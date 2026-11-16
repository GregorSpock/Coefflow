import pandas as pd


# Every port has a type saying what travels through it, so a line can only be drawn between
# ports that suit each other and a module is never handed something it can't use:
#   "table"  a table of data, which is what most ports carry
#   "model"  the coefficient table of a fitted model, carrying the details of the fit that
#            Forecast, Hausman and Reg Table need. A model is still a table, so a model output
#            can go into any table input, but a table can't go into an input that needs a model
# A module says what its ports carry with two lists, one entry per port in port order:
#   OUTPUT_TYPES = ["model", "table"]
#   INPUT_TYPES = ["model", "table"]
# An input that takes more than one type gives them as a tuple, e.g. ("model", "table"), the
# first being the one the port is drawn as. A module with neither list, like every module
# written before ports had types, has table ports throughout. A growing node's inputs all have
# the type of its first
TABLE = "table"
MODEL = "model"

# What each type is called when a connection is refused
NAMES = {
    TABLE: "a table",
    MODEL: "a fitted model (Out 1 of a regression)",
}


# The type a node's output port gives
def output_type(node, index):
    types = getattr(node.module, "OUTPUT_TYPES", [])
    return types[index] if index < len(types) else TABLE


# The types a node's input port takes, as a tuple with the one it is drawn as first
def input_types(node, index):
    types = getattr(node.module, "INPUT_TYPES", [])
    if getattr(node, "growing", False):
        index = 0
    accepted = types[index] if index < len(types) else TABLE
    return (accepted,) if isinstance(accepted, str) else tuple(accepted)


# True if an input taking the accepted types can be given something of the given type
def suits(given, accepted):
    return given in accepted or (given == MODEL and TABLE in accepted)


# Why a source's output can't go into a target's input, or None if it can
def mismatch(source, output_index, target, input_index):
    given = output_type(source, output_index)
    accepted = input_types(target, input_index)
    if suits(given, accepted):
        return None
    wanted = " or ".join(NAMES.get(kind, kind) for kind in accepted)
    return (
        f"In {input_index + 1} of {target.name} takes {wanted}, "
        f"but Out {output_index + 1} of {source.name} gives {NAMES.get(given, given)}"
    )


# True if a table carries the details of a fitted model
def is_model(value):
    return isinstance(value, pd.DataFrame) and ("fit" in value.attrs or "model" in value.attrs)


# Why a value isn't what a port of the given type carries, or None if it is
def problem_with(value, kind):
    if not isinstance(value, pd.DataFrame):
        return f"is {type(value).__name__}, not a table"
    if kind == MODEL and not is_model(value):
        return "is a table without the details of a fitted model"
    return None


# Checks what a node is about to be given on each connected input, raising ValueError with
# the reason if an input that needs a model has been given a plain table
def check_inputs(node, inputs):
    for index, value in enumerate(inputs):
        accepted = input_types(node, index)
        if value is None or TABLE in accepted:
            continue
        if not any(problem_with(value, kind) is None for kind in accepted):
            raise ValueError(
                f"In {index + 1} needs {NAMES[accepted[0]]}, connected directly. What arrived {problem_with(value, accepted[0])}"
            )


# Checks what a node gave back: a list with something of the right type for every output port.
# A failure here is a fault in the module rather than in the workflow, and the message says so
def check_outputs(node, outputs):
    if not isinstance(outputs, (list, tuple)):
        raise TypeError(f"The module's run function gave back {type(outputs).__name__} rather than a list of outputs")
    if len(outputs) < node.outputs:
        raise TypeError(f"The module's run function gave {len(outputs)} outputs, but it has {node.outputs} output ports")
    for index in range(node.outputs):
        problem = problem_with(outputs[index], output_type(node, index))
        if problem:
            raise TypeError(f"Out {index + 1} {problem}. This is a fault in the module")
