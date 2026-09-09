"""One focused regression for the analyzer's only nontrivial parser.

The live runner serialised its timing/counter columns with ``str(dict)``, so
those CSV cells are Python literals rather than JSON. Everything else in the
analyzer is checked by its own real-data self-checks (the install-AoI
median/p95 cross-check against the registered values, the counter-reconciliation
and reassembly identities, and the row/uniqueness assertions), so only
``parse_pydict`` is unit-tested here.
"""

from consolidate_288 import parse_pydict


def test_parses_repr_serialised_timing_blob():
    text = (
        "{'front_backbone': 1095491, 'ranker_selection': 2558849, "
        "'ae_encode': 1204085, 'total_ue_preparation': 8837490}"
    )
    assert parse_pydict(text) == {
        "front_backbone": 1095491,
        "ranker_selection": 2558849,
        "ae_encode": 1204085,
        "total_ue_preparation": 8837490,
    }


def test_absent_or_unparseable_yields_empty_so_caller_counts_it_unavailable():
    # Blank/None must not be guessed at; they are counted as unavailable.
    assert parse_pydict("") == {}
    assert parse_pydict("   ") == {}
    assert parse_pydict(None) == {}
    # Malformed input must not raise and must not be partially salvaged.
    assert parse_pydict("{'front_backbone': }") == {}
    assert parse_pydict("not a dict at all") == {}
    # A well-formed literal that is not a dict is still unusable.
    assert parse_pydict("[1, 2, 3]") == {}
    assert parse_pydict("42") == {}


def test_dict_passthrough_is_not_restringified():
    payload = {"frozen_tail": 83958484}
    assert parse_pydict(payload) is payload
