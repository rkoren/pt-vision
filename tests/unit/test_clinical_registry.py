import pytest

import ptvision.clinical as clinical
from ptvision.clinical.metrics.base import Metric
from ptvision.clinical.norms import compare, load_norms
from ptvision.clinical.protocol import builtin_protocol_ids, load_protocol
from ptvision.data.models import Subject


def test_protocols_reference_registered_names_and_norms() -> None:
    norms = load_norms()
    ids = builtin_protocol_ids()
    assert "sts_5x" in ids and "sts_30s" in ids
    for pid in ids:
        p = load_protocol(pid)
        assert p.id == pid
        clinical.registry.get_segmenter(p.segmenter.name)
        assert p.primary_metric() is not None
        for m in p.metrics:
            clinical.registry.get_metric(m.name)
            if m.norm:
                assert m.norm in norms
                assert norms[m.norm].metric == m.name
        assert p.source_sha256


def test_tier1_metrics_have_citations_and_versions() -> None:
    for e in clinical.registry.METRICS.values():
        assert e.version
        if e.tier == 1:
            assert e.citation


def test_norm_comparison_statements() -> None:
    m = Metric(
        name="sts_total_time",
        label="5xSTS",
        value=13.1,
        units="s",
        error=0.07,
        error_kind="resolution",
        tier=1,
        method_version="0.1.0",
    )
    c = compare(m, "sts_5x_bohannon2006", Subject(age_years=72, sex="f"))
    assert c.applicable and c.reference_value == 12.6
    assert "slower than" in c.statement and "70–79" in c.statement
    c2 = compare(m, "sts_5x_bohannon2006", Subject(age_years=45))
    assert not c2.applicable and "No reference band for age 45" in c2.statement
    c3 = compare(m, "sts_5x_bohannon2006", Subject())
    assert not c3.applicable and "age not recorded" in c3.statement
    fast = m.model_copy(update={"value": 9.0})
    assert (
        "faster than or equal to"
        in compare(fast, "sts_5x_bohannon2006", Subject(age_years=65)).statement
    )


def test_unknown_protocol_raises() -> None:
    with pytest.raises(KeyError):
        load_protocol("nope")
