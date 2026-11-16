import numpy as np
import pandas as pd
import pytest

from engine import dates
from engine.node import load_module
from tests.support import SAMPLE_DATA


# The data preparation modules: Group By, Pivot, Concat, Missing Values and Dates, each checked
# against the same result worked out directly in pandas

DATA = pd.read_csv(SAMPLE_DATA)
group_by = load_module("Group By")
pivot = load_module("Pivot")
concat = load_module("Concat")
missing = load_module("Missing Values")
dates_module = load_module("Dates")

LONG = pd.DataFrame({
    "id": [1, 1, 2, 2],
    "year": [2019, 2020, 2019, 2020],
    "gdp": [1.0, 2.0, 3.0, 4.0],
    "pop": [5, 6, 7, 8],
})


# ----- Group By -----

def test_group_by_gives_a_row_per_group():
    result = group_by.run([DATA], {"groups": ["region"], "statistics": {"unemployment": ["mean", "sd", "n", "max"]}})[0]
    expected = DATA.groupby("region")["unemployment"]
    assert result["region"].tolist() == sorted(DATA["region"].unique())
    assert np.allclose(result["unemployment_mean"], expected.mean().to_numpy())
    assert np.allclose(result["unemployment_sd"], expected.std().to_numpy())
    assert result["rows"].tolist() == expected.size().tolist()
    assert result["unemployment_max"].tolist() == expected.max().tolist()


def test_group_by_with_no_groups_summarises_the_whole_table():
    result = group_by.run([DATA], {"statistics": {"gdp_growth": ["median", "distinct"]}})[0]
    assert len(result) == 1 and result["rows"].iloc[0] == len(DATA)
    assert result["gdp_growth_median"].iloc[0] == DATA["gdp_growth"].median()
    assert result["gdp_growth_distinct"].iloc[0] == DATA["gdp_growth"].nunique()


def test_group_by_on_every_row_adds_the_group_statistics():
    result = group_by.run([DATA], {"groups": ["region", "urban"], "statistics": {"inflation": ["mean"]}, "every_row": True})[0]
    assert len(result) == len(DATA)
    assert np.allclose(result["inflation_mean"], DATA.groupby(["region", "urban"])["inflation"].transform("mean"))


def test_group_by_keeps_missing_groups_and_refuses_the_mean_of_text():
    data = pd.DataFrame({"g": ["a", None, "a"], "x": [1.0, 2.0, 3.0]})
    result = group_by.run([data], {"groups": ["g"], "statistics": {"x": ["sum"]}})[0]
    assert result["x_sum"].sum() == 6.0 and len(result) == 2
    with pytest.raises(ValueError, match="not numeric"):
        group_by.run([DATA], {"groups": ["year"], "statistics": {"region": ["mean"]}})


# ----- Pivot -----

def test_pivot_spreads_long_to_wide_and_back():
    wide = pivot.run([LONG], {"direction": "Long to wide", "ids": ["id"], "column": "year", "values": ["gdp", "pop"]})[0]
    assert wide.columns.tolist() == ["id", "gdp_2019", "gdp_2020", "pop_2019", "pop_2020"]
    assert wide.loc[wide["id"] == 2, "gdp_2020"].item() == 4.0

    back = pivot.run([wide], {"direction": "Wide to long", "ids": ["id"], "values": [], "stubs": True})[0]
    assert back.columns.tolist() == ["id", "period", "gdp", "pop"]
    merged = back.merge(LONG, left_on=["id", "period"], right_on=["id", "year"])
    assert len(merged) == 4 and (merged["gdp_x"] == merged["gdp_y"]).all() and (merged["pop_x"] == merged["pop_y"]).all()


def test_pivot_combines_duplicates_as_chosen():
    doubled = pd.concat([LONG, LONG.assign(gdp=LONG["gdp"] * 3)])
    wide = pivot.run([doubled], {"direction": "Long to wide", "ids": ["id"], "column": "year", "values": ["gdp"], "combine": "mean"})[0]
    assert wide.loc[wide["id"] == 1, "gdp_2019"].item() == 2.0
    assert "combined" in wide.attrs["note"]


def test_pivot_stacks_fields_into_one_column():
    long = pivot.run([DATA.head(3)], {"direction": "Wide to long", "ids": ["region", "year"], "values": ["inflation", "bank_rate"], "names": "measure, amount"})[0]
    assert long.columns.tolist() == ["region", "year", "measure", "amount"]
    assert len(long) == 6 and set(long["measure"]) == {"inflation", "bank_rate"}


def test_pivot_split_names():
    assert pivot.split_name("gdp_2019") == ("gdp", "2019")
    assert pivot.split_name("pop1990") == ("pop", "1990")
    assert pivot.split_name("growth_rate_q1") == ("growth_rate", "q1")
    assert pivot.split_name("plain") is None


# ----- Concat -----

def test_concat_stacks_rows_in_input_order():
    first, second = DATA.head(3), DATA.tail(2)[["year", "region", "inflation"]]
    result = concat.run([first, second, None], {"source": True})[0]
    assert len(result) == 5 and result["source"].tolist() == [1, 1, 1, 2, 2]
    assert result.columns.tolist()[:3] == ["source", "region", "year"]
    assert result["gdp_growth"].iloc[3:].isna().all()

    shared = concat.run([first, second], {"fields": "Shared fields"})[0]
    assert shared.columns.tolist() == ["region", "year", "inflation"]


def test_concat_can_drop_repeated_rows():
    result = concat.run([LONG, LONG], {"dedupe": True})[0]
    assert result.equals(LONG)


# ----- Missing Values -----

GAPS = pd.DataFrame({
    "group": ["a", "a", "a", "b", "b", "b"],
    "x": [1.0, np.nan, 3.0, 10.0, np.nan, np.nan],
    "label": ["p", None, "p", "q", "r", None],
})


@pytest.mark.parametrize("method, group, expected", [
    ("mean", None, [1, 14 / 3, 3, 10, 14 / 3, 14 / 3]),
    ("mean", "group", [1, 2, 3, 10, 10, 10]),
    ("median", None, [1, 3, 3, 10, 3, 3]),
    ("value", None, [1, -1, 3, 10, -1, -1]),
    ("prev", "group", [1, 1, 3, 10, 10, 10]),
    ("next", None, [1, 3, 3, 10, np.nan, np.nan]),
    ("interp", None, [1, 2, 3, 10, np.nan, np.nan]),
])
def test_missing_values_are_filled(method, group, expected):
    result = missing.run([GAPS], {"methods": {"x": method}, "group": group, "value": "-1"})[0]
    assert np.allclose(result["x"].to_numpy(float), np.array(expected, dtype=float), equal_nan=True)


def test_missing_values_flags_drops_and_reports():
    result = missing.run([GAPS], {"methods": {"x": "mean", "label": "drop"}, "flag": True})[0]
    assert result["x_missing"].tolist() == [0, 0, 0, 1]
    assert len(result) == 4
    assert "dropped 2 rows" in result.attrs["summary"]
    filled = missing.run([GAPS], {"methods": {"label": "mode"}, "group": "group"})[0]
    assert filled["label"].tolist() == ["p", "p", "p", "q", "r", "q"]
    with pytest.raises(ValueError, match="not numeric"):
        missing.run([GAPS], {"methods": {"label": "mean"}})
    with pytest.raises(ValueError, match="not a number"):
        missing.run([GAPS], {"methods": {"x": "value"}, "value": "abc"})


# ----- Dates -----

@pytest.mark.parametrize("text, expected", [
    ("2020Q3", "2020-07-01"), ("2020-Q1", "2020-01-01"), ("Q4 2019", "2019-10-01"), ("2020 q2", "2020-04-01"),
    ("2020M03", "2020-03-01"), ("2020m11", "2020-11-01"), ("1990:1", "1990-01-01"), ("1990:3", "1990-07-01"),
    ("1990:03", "1990-03-01"), ("1990:12", "1990-12-01"), ("1987", "1987-01-01"),
])
def test_period_labels_are_read(text, expected):
    assert dates.period_start(text) == pd.Timestamp(expected)


def test_dates_are_parsed_in_many_forms():
    assert dates.parse(pd.Series(["2021-03-31", "2021-06-30"])).dt.month.tolist() == [3, 6]
    assert dates.parse(pd.Series(["03/04/2020"]), dayfirst=True).iloc[0] == pd.Timestamp("2020-04-03")
    assert dates.parse(pd.Series(["03/04/2020"])).iloc[0] == pd.Timestamp("2020-03-04")
    assert dates.parse(pd.Series(["31.12.2020"]), fmt="%d.%m.%Y").iloc[0] == pd.Timestamp("2020-12-31")
    assert dates.parse(pd.Series([1995, 1996])).dt.year.tolist() == [1995, 1996]
    assert dates.parse(pd.Series(["2020-01-01", "nonsense"])).isna().tolist() == [False, True]
    with pytest.raises(ValueError, match="format"):
        dates.parse(pd.Series([20200131, 20200229]))
    assert dates.parse(pd.Series([20200131]), fmt="%Y%m%d").iloc[0] == pd.Timestamp("2020-01-31")


def test_dates_module_adds_parts_gaps_and_a_trend():
    data = pd.DataFrame({"start": ["2020Q1", "2020Q3", None], "end": ["2021-03-31", "2021-06-30", "bad"], "year": [1990, 1992, 1993]})
    settings = {
        "convert": ["start"], "parts": {"end": ["Year", "Qtr", "Month", "Wkday", "YQ", "YM"]},
        "start": "start", "end": "end", "trend": "year", "unit": "months",
    }
    result = dates_module.run([data], settings)[0]
    assert pd.api.types.is_datetime64_any_dtype(result["start"])
    assert result["end_year"].tolist()[:2] == [2021, 2021] and pd.isna(result["end_year"].iloc[2])
    assert result["end_quarter"].tolist()[:2] == [1, 2]
    assert result["end_weekday"].tolist()[:2] == [3, 3]
    assert result["end_yq"].tolist()[:2] == ["2021Q1", "2021Q2"]
    assert result["end_ym"].tolist()[:2] == ["2021-03", "2021-06"]
    assert result["months_between"].tolist()[:2] == [14, 11]
    assert result["trend_months"].tolist() == [0, 24, 36]
    assert "1 values of end" in result.attrs["unread"]


def test_dates_module_needs_both_ends():
    with pytest.raises(ValueError, match="both"):
        dates_module.run([LONG], {"start": "year"})


# Renaming a field upstream follows into the new modules' settings
def test_new_modules_follow_renamed_fields():
    from engine.fields import update_settings
    from engine.node import Node
    for name, settings, key in [
        ("Group By", {"groups": ["region"], "statistics": {"region": ["n"]}}, "groups"),
        ("Pivot", {"ids": ["region"], "column": "region", "values": []}, "ids"),
        ("Missing Values", {"methods": {"region": "mode"}, "group": "region"}, "methods"),
        ("Dates", {"convert": ["region"], "parts": {}, "trend": "region"}, "convert"),
    ]:
        node = Node(name, 0, 0, 1)
        node.settings = settings
        update_settings(node, "region", "area")
        assert "region" not in str(node.settings[key]) and "area" in str(node.settings[key]), name
