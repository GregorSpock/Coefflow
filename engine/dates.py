import re
import warnings

import numpy as np
import pandas as pd


# Turning the many ways dates are written into real dates, for the Dates module and anything else
# that needs a time field. Besides everything pandas reads (2020-03-31, 31/03/2020, March 2020...)
# it reads the period labels economists' data comes with:
#   2020          a plain year, taken as 1 January
#   2020Q1        quarters, also 2020-Q1, 2020 Q1, 2020q1 and gretl's 2020:1
#   2020M03       months, also 2020-M3, 2020m3 and gretl's 2020:03
# A quarter or month is taken as its first day. Text that can't be read becomes missing

QUARTER = re.compile(r"^\s*(\d{4})\s*[-:/ ]?\s*[Qq]\s*([1-4])\s*$")
QUARTER_FIRST = re.compile(r"^\s*[Qq]([1-4])\s*[-:/ ]?\s*(\d{4})\s*$")
MONTH = re.compile(r"^\s*(\d{4})\s*[-:/ ]?\s*[Mm]\s*(\d{1,2})\s*$")
GRETL = re.compile(r"^\s*(\d{4}):(\d{1,2})\s*$")
YEAR = re.compile(r"^\s*(\d{4})(?:\.0+)?\s*$")


# The first day of a period label, or None if the text isn't one
def period_start(text):
    text = str(text)
    for pattern, order in [(QUARTER, "yq"), (QUARTER_FIRST, "qy")]:
        match = pattern.match(text)
        if match:
            year, quarter = (match.group(1), match.group(2)) if order == "yq" else (match.group(2), match.group(1))
            return pd.Timestamp(int(year), 3 * int(quarter) - 2, 1)
    match = MONTH.match(text)
    if match and 1 <= int(match.group(2)) <= 12:
        return pd.Timestamp(int(match.group(1)), int(match.group(2)), 1)
    match = GRETL.match(text)
    if match:
        year, period = int(match.group(1)), match.group(2)
        # gretl writes months with two digits (1990:01) and quarters with one (1990:1)
        if len(period) == 1 and 1 <= int(period) <= 4:
            return pd.Timestamp(year, 3 * int(period) - 2, 1)
        if 1 <= int(period) <= 12:
            return pd.Timestamp(year, int(period), 1)
    match = YEAR.match(text)
    if match and 1000 <= int(match.group(1)) <= 9999:
        return pd.Timestamp(int(match.group(1)), 1, 1)
    return None


# A field as dates. fmt is a strftime pattern such as %d/%m/%Y, or "" to work it out.
# dayfirst reads 03/04/2020 as 3 April rather than March 4. Whole numbers between 1000 and 9999
# are taken as years. Returns the dates, with what couldn't be read left missing
def parse(values, fmt="", dayfirst=False):
    if pd.api.types.is_datetime64_any_dtype(values):
        return values
    if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
        whole = values.dropna()
        if len(whole) and (whole == whole.round()).all() and whole.between(1000, 9999).all():
            return pd.to_datetime(values.astype("Int64").astype(str).where(values.notna()), format="%Y", errors="coerce")
        if fmt:
            return pd.to_datetime(values.astype("Int64").astype(str).where(values.notna()), format=fmt, errors="coerce")
        raise ValueError(f"{values.name} holds numbers that aren't years, so give its format, e.g. %Y%m%d")

    text = values.astype("string")
    if fmt:
        return pd.to_datetime(text, format=fmt, errors="coerce")

    sample = text.dropna().head(50)
    periods = sample.map(period_start)
    if len(sample) and periods.notna().all():
        return text.map(lambda value: period_start(value) if pd.notna(value) else None).astype("datetime64[ns]")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parsed = pd.to_datetime(text, dayfirst=dayfirst, errors="coerce", format="mixed")
        except (TypeError, ValueError):
            parsed = pd.to_datetime(text, dayfirst=dayfirst, errors="coerce")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed


# The parts a date can be split into: the button's text, the new field's ending and how to work it out
PARTS = {
    "Year": ("year", lambda d: d.dt.year),
    "Qtr": ("quarter", lambda d: d.dt.quarter),
    "Month": ("month", lambda d: d.dt.month),
    "Day": ("day", lambda d: d.dt.day),
    "Wkday": ("weekday", lambda d: d.dt.dayofweek + 1),
    "Week": ("week", lambda d: d.dt.isocalendar().week.astype("Int64")),
    "YQ": ("yq", lambda d: d.dt.to_period("Q").astype(str).where(d.notna())),
    "YM": ("ym", lambda d: d.dt.to_period("M").astype(str).where(d.notna())),
}


# A part of each date as a new field: whole numbers stay whole even with missing dates
def part(dates, name):
    values = PARTS[name][1](dates)
    if pd.api.types.is_float_dtype(values):
        return values.astype("Int64")
    return values


# Days, weeks, months or years from one date to another, as a number
def between(start, end, unit):
    if unit == "days":
        return (end - start).dt.days.astype("Int64")
    if unit == "weeks":
        return (end - start).dt.days / 7
    if unit == "months":
        months = (end.dt.year - start.dt.year) * 12 + (end.dt.month - start.dt.month)
        return months.astype("Int64")
    if unit == "years":
        return ((end - start).dt.days / 365.25)
    raise ValueError(f"Unknown unit: {unit}")


# A count of periods since the earliest date, 0 for it, in the unit given: a time trend for a model
def trend(dates, unit):
    first = dates.min()
    if pd.isna(first):
        return pd.Series(np.nan, index=dates.index)
    return between(pd.Series(first, index=dates.index), dates, unit)
