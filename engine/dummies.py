import numpy as np
import pandas as pd


# Lets the regression modules take text fields (e.g. region) as X or C variables.
# A text field enters as one 0/1 dummy per category except the first, which is the baseline
# the other categories' coefficients are measured against. Y must still be numeric


# The fields that can take a role in a regression: numeric ones and text ones, but not dates
def model_columns(data):
    return [c for c in data.columns if not pd.api.types.is_datetime64_any_dtype(data[c])]


def is_numeric(values):
    return pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values)


# The columns a field adds to the model: itself if numeric, else its dummies,
# named like region=London
def terms(values, column):
    if is_numeric(values):
        return values.astype(float).to_frame(column)
    part = pd.get_dummies(values.astype(str), prefix=column, prefix_sep="=", drop_first=True, dtype=float)
    if part.empty:
        raise ValueError(f"{column} has only one value in the rows used, so it adds nothing to the model")
    return part


# The X and C columns for a fit, from (fields, role) pairs, and the role of each column
def design_matrix(used, groups):
    parts, roles = [], []
    for columns, role in groups:
        for column in columns:
            part = terms(used[column], column)
            parts.append(part)
            roles += [role] * part.shape[1]
    if not parts:
        return pd.DataFrame(index=used.index), roles
    return pd.concat(parts, axis=1), roles


# Stops a fit whose columns are perfectly collinear, such as a field that never changes within a
# region alongside region dummies. statsmodels would otherwise split the effect arbitrarily and
# still report tiny standard errors. The error names every field caught up in it
def check_full_rank(x):
    values = x.to_numpy(dtype=float)
    rank = np.linalg.matrix_rank(values)
    if rank == values.shape[1]:
        return

    fields = []
    for i, column in enumerate(x.columns):
        if column != "const" and np.linalg.matrix_rank(np.delete(values, i, axis=1)) == rank:
            fields.append(str(column).split("=")[0])
    fields = list(dict.fromkeys(fields)) or ["the intercept"]
    raise ValueError(
        f"These variables are perfectly collinear, so their effects can't be told apart: "
        f"{', '.join(fields)}. Remove one of them"
    )
