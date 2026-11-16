import io
import json
import os
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from engine.panel import Panel


INPUTS = 0
OUTPUTS = 1
INPUT_SPECS = []
OUTPUT_SPECS = [
    ("Data", "the series downloaded, in long form (one row per series and period with series, name, period and value, plus "
             "country for the World Bank) or wide form (one row per period and a column per series)"),
]

SOURCES = [("fred", "FRED"), ("worldbank", "World Bank"), ("oecd", "OECD"), ("dbnomics", "DBnomics")]
SHAPES = [("long", "Long"), ("wide", "Wide")]
# What to type for each source, shown as the text box's label, and an example used as the default
EXAMPLES = {
    "fred": ("FRED series, e.g. UNRATE, CPIAUCSL; from=2000-01-01", "UNRATE, CPIAUCSL; from=2000-01-01"),
    "worldbank": ("World Bank indicators; countries, e.g. NY.GDP.MKTP.KD.ZG; countries=GBR,USA; from=2000",
                  "NY.GDP.MKTP.KD.ZG; countries=GBR,USA,DEU; from=2000"),
    "oecd": ("OECD dataflow/key (several apart by spaces), e.g. OECD.SDD.STES,DSD_STES@DF_CLI,/USA+GBR.M.LI...AA...H; from=2015-01",
             "OECD.SDD.STES,DSD_STES@DF_CLI,/USA+GBR.M.LI...AA...H; from=2015-01"),
    "dbnomics": ("DBnomics series ids, e.g. AMECO/ZUTN/EA19.1.0.0.0.ZUTN; from=2000", "AMECO/ZUTN/EA19.1.0.0.0.ZUTN; from=2000"),
}
TIMEOUT = 60
# What the server is told about who is asking: the app's name, nothing about the user
USER_AGENT = "Coefflow"
FIELD_SETTINGS = []

# Downloads made this session by address, so opening the panel again doesn't fetch again.
# Refresh fetches anew
cache = {}


def set_defaults(settings):
    settings.setdefault("source", "fred")
    settings.setdefault("shape", "long")
    settings.setdefault("query", EXAMPLES[settings["source"]][1])
    settings.setdefault("refresh", 0)


def on_click(node):
    set_defaults(node.settings)
    panel = Panel(node, "Online Data")
    panel.add_button(label(node.settings, "source", SOURCES, "Source"), cycle_source)
    panel.add_button(label(node.settings, "shape", SHAPES, "Shape"), cycle_shape)
    panel.add_button("Example", use_example)
    panel.add_button("Refresh", refresh)
    panel.add_text_field(EXAMPLES[node.settings["source"]][0], node.settings["query"], type_query, blocked="", fill=True)
    panel.preview_title = "Data"
    panel.refresh = show
    show(panel)
    return panel


def label(settings, key, options, name):
    return f"{name}: {dict(options)[settings[key]]}"


def cycle_source(panel, button):
    settings = panel.node.settings
    values = [v for v, t in SOURCES]
    settings["source"] = values[(values.index(settings["source"]) + 1) % len(values)]
    settings["query"] = EXAMPLES[settings["source"]][1]
    button.text = label(settings, "source", SOURCES, "Source")
    panel.add_text_field(EXAMPLES[settings["source"]][0], settings["query"], type_query, blocked="", fill=True)
    show(panel)


def cycle_shape(panel, button):
    settings = panel.node.settings
    settings["shape"] = "wide" if settings["shape"] == "long" else "long"
    button.text = label(settings, "shape", SHAPES, "Shape")
    show(panel)


def use_example(panel, button):
    settings = panel.node.settings
    settings["query"] = EXAMPLES[settings["source"]][1]
    panel.text_field.set_text(settings["query"])
    show(panel)


# Changes a setting that does nothing else, so the next run counts as changed and downloads again
def refresh(panel, button):
    panel.node.settings["refresh"] = panel.node.settings.get("refresh", 0) + 1
    cache.clear()
    show(panel)


def type_query(panel, text):
    panel.node.settings["query"] = text
    show(panel)


def show(panel):
    node = panel.node
    set_defaults(node.settings)
    panel.preview = None
    panel.note = ""
    panel.lines = [f"Source: {dict(SOURCES)[node.settings['source']]}    Shape: {node.settings['shape']}",
                   "Downloads happen when the workflow runs; Refresh downloads again"]
    if node.last_outputs and isinstance(node.last_outputs[0], pd.DataFrame):
        data = node.last_outputs[0]
        panel.lines += data.attrs.get("summary", [])
        panel.note = " ".join(data.attrs.get("notes", []))
        panel.preview = data
        panel.message = ""
    else:
        panel.message = node.error or "Type the series to download, then run the workflow"


# Reads "codes; name=value; name=value" into the list of codes and a dict of options. Codes are
# separated by commas, or by spaces for the OECD, whose codes have commas of their own
def parse_query(text, spaces=False):
    parts = [p.strip() for p in str(text or "").split(";")]
    first = parts[0] if parts else ""
    codes = [c.strip() for c in (first.split() if spaces else first.split(",")) if c.strip()]
    options = {}
    for part in parts[1:]:
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Write options as name=value after a semicolon, e.g. from=2000, not {part}")
        name, value = [p.strip() for p in part.split("=", 1)]
        options[name.lower()] = value
    if not codes:
        raise ValueError("Type at least one series code")
    return codes, options


# Downloads an address, giving a plain message if the connection or server fails
def fetch(url):
    if url in cache:
        return cache[url]
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            text = response.read().decode("utf-8-sig")
    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise ValueError("The server has no data for that code (404). Check the code is spelled as the source writes it")
        raise ValueError(f"The server refused the request ({error.code} {error.reason})")
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        reason = getattr(error, "reason", error)
        raise ValueError(f"Couldn't reach the server ({reason}). Check the internet connection and try again")
    cache[url] = text
    return text


# Values as numbers, with the sources' marks for a missing value (".", "NA", blank) as missing
def to_number(values):
    values = pd.Series(values, dtype=object)
    return pd.to_numeric(values.where(~values.isin([".", "NA", ""])), errors="coerce")


# FRED, the St Louis Fed's database. With a free API key in the FRED_API_KEY environment variable
# the official API is used; without one, the CSV that FRED's own graphs download
def read_fred(codes, options):
    key = os.environ.get("FRED_API_KEY")
    frames = []
    for code in codes:
        if key:
            query = {"series_id": code, "api_key": key, "file_type": "json"}
            if options.get("from"):
                query["observation_start"] = options["from"]
            if options.get("to"):
                query["observation_end"] = options["to"]
            data = json.loads(fetch("https://api.stlouisfed.org/fred/series/observations?" + urllib.parse.urlencode(query)))
            observations = data.get("observations", [])
            frame = pd.DataFrame({"period": [o["date"] for o in observations], "value": to_number([o["value"] for o in observations])})
        else:
            query = {"id": code}
            if options.get("from"):
                query["cosd"] = options["from"]
            if options.get("to"):
                query["coed"] = options["to"]
            text = fetch("https://fred.stlouisfed.org/graph/fredgraph.csv?" + urllib.parse.urlencode(query))
            frame = parse_fred_csv(text, code)
        frame.insert(0, "series", code)
        frame.insert(1, "name", code)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def parse_fred_csv(text, code):
    table = pd.read_csv(io.StringIO(text), dtype=str)
    if table.empty or len(table.columns) < 2:
        raise ValueError(f"FRED sent nothing for {code}")
    date_column = table.columns[0]
    value_column = code if code in table.columns else table.columns[1]
    return pd.DataFrame({"period": table[date_column], "value": to_number(table[value_column]).to_numpy()})


# The World Bank's indicators API: one row per country and year
def read_worldbank(codes, options):
    countries = options.get("countries", "all").replace(",", ";").replace(" ", "")
    frames = []
    for code in codes:
        query = {"format": "json", "per_page": 20000}
        if options.get("from") or options.get("to"):
            query["date"] = f"{options.get('from', '1960')}:{options.get('to', '2100')}"
        url = f"https://api.worldbank.org/v2/country/{urllib.parse.quote(countries, safe=';')}/indicator/{urllib.parse.quote(code)}?" + urllib.parse.urlencode(query)
        frames.append(parse_worldbank(fetch(url), code))
    return pd.concat(frames, ignore_index=True)


def parse_worldbank(text, code):
    data = json.loads(text)
    if not isinstance(data, list) or len(data) < 2 or data[1] is None:
        message = data[0].get("message", [{}])[0].get("value") if isinstance(data, list) and data and isinstance(data[0], dict) else None
        raise ValueError(f"The World Bank has no data for {code}" + (f": {message}" if message else ""))
    rows = data[1]
    return pd.DataFrame({
        "series": code,
        "name": [r["indicator"]["value"] for r in rows],
        "country": [r.get("countryiso3code") or r["country"]["id"] for r in rows],
        "country_name": [r["country"]["value"] for r in rows],
        "period": [int(r["date"]) if str(r["date"]).isdigit() else r["date"] for r in rows],
        "value": to_number([r["value"] for r in rows]).to_numpy(),
    }).sort_values(["country", "period"], kind="stable").reset_index(drop=True)


# The OECD's SDMX data API, as CSV with labels. A code is the dataflow and key, as the OECD Data
# Explorer shows them under Developer API
def read_oecd(codes, options):
    frames = []
    for code in codes:
        if "/" not in code:
            raise ValueError("Write an OECD code as dataflow/key, e.g. OECD.SDD.STES,DSD_STES@DF_CLI,/USA.M.LI...AA...H")
        flow, key = code.split("/", 1)
        query = {"format": "csvfilewithlabels"}
        if options.get("from"):
            query["startPeriod"] = options["from"]
        if options.get("to"):
            query["endPeriod"] = options["to"]
        url = f"https://sdmx.oecd.org/public/rest/data/{urllib.parse.quote(flow, safe=',@')}/{urllib.parse.quote(key, safe='.+')}?" + urllib.parse.urlencode(query)
        frames.append(parse_oecd(fetch(url), code))
    return pd.concat(frames, ignore_index=True)


def parse_oecd(text, code):
    table = pd.read_csv(io.StringIO(text), dtype=str)
    if "TIME_PERIOD" not in table or "OBS_VALUE" not in table:
        raise ValueError(f"The OECD sent no observations for {code}")
    # Dimensions are the coded columns that vary between rows, e.g. REF_AREA; their labelled twins follow them
    skip = {"STRUCTURE", "STRUCTURE_ID", "STRUCTURE_NAME", "ACTION", "TIME_PERIOD", "OBS_VALUE", "Time period", "Observation value"}
    columns = list(table.columns)
    dimensions = [c for i, c in enumerate(columns) if c.isupper() and c not in skip and table[c].nunique() > 1]
    output = pd.DataFrame(index=table.index)
    output["series"] = table[dimensions].agg(".".join, axis=1) if dimensions else code
    output["name"] = table.get("STRUCTURE_NAME", code)
    for column in dimensions:
        output[column.lower()] = table[column]
    output["period"] = table["TIME_PERIOD"]
    output["value"] = to_number(table["OBS_VALUE"]).to_numpy()
    return output.sort_values(["series", "period"], kind="stable").reset_index(drop=True)


# DBnomics, which gathers series from many statistics offices and central banks under one address
def read_dbnomics(codes, options):
    query = {"series_ids": ",".join(codes), "observations": 1, "format": "json"}
    text = fetch("https://api.db.nomics.world/v22/series?" + urllib.parse.urlencode(query))
    table = parse_dbnomics(text)
    if options.get("from"):
        table = table[table["period"].astype(str) >= options["from"]]
    if options.get("to"):
        table = table[table["period"].astype(str) <= options["to"]]
    return table.reset_index(drop=True)


def parse_dbnomics(text):
    data = json.loads(text)
    docs = (data.get("series") or {}).get("docs") or []
    if not docs:
        message = data.get("message") or "no series found"
        raise ValueError(f"DBnomics sent no series: {message}")
    frames = []
    for doc in docs:
        code = f"{doc.get('provider_code')}/{doc.get('dataset_code')}/{doc.get('series_code')}"
        frames.append(pd.DataFrame({"series": code, "name": doc.get("series_name", code), "period": doc.get("period", []),
                                    "value": to_number(doc.get("value", [])).to_numpy()}))
    return pd.concat(frames, ignore_index=True)


READERS = {"fred": read_fred, "worldbank": read_worldbank, "oecd": read_oecd, "dbnomics": read_dbnomics}


# One row per period and a column per series (per series and country for the World Bank)
def widen(table):
    keys = ["country", "country_name", "period"] if "country" in table else ["period"]
    names = table["series"].astype(str)
    wide = table.assign(series=names).pivot_table(index=keys, columns="series", values="value", aggfunc="first").reset_index()
    wide.columns.name = None
    return wide


# Downloads the typed series from the chosen source as a table
def run(inputs, settings):
    set_defaults(settings)
    codes, options = parse_query(settings["query"], spaces=settings["source"] == "oecd")
    table = READERS[settings["source"]](codes, options)
    notes = []
    if table.empty:
        raise ValueError("The download worked but holds no observations; widen the dates or check the codes")
    missing = int(table["value"].isna().sum())
    if missing:
        notes.append(f"{missing} observations are missing in the source.")
    if settings["source"] == "fred" and not os.environ.get("FRED_API_KEY"):
        notes.append("Set a free FRED API key in the FRED_API_KEY environment variable to use FRED's official API.")
    summary = [f"{table['series'].nunique()} series, {len(table)} observations, "
               f"{table['period'].astype(str).min()} to {table['period'].astype(str).max()}"]
    if settings["shape"] == "wide":
        table = widen(table)
    table.attrs = {"summary": summary, "notes": notes}
    return [table]
