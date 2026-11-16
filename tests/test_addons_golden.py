import pytest

from tests.addon_cases import CASES, run_case
from tests.support import read_golden, same, summarise_outputs


GOLDEN = read_golden("addons")


# Every add-on module still gives the results it gave when the golden summaries were made
@pytest.mark.parametrize("name", sorted(CASES))
def test_addon_matches_golden(name):
    assert same(summarise_outputs(run_case(name)), GOLDEN[name])
