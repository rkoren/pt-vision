import pytest
from pydantic import ValidationError

from ptvision.clinical.protocol import RuleSpec, load_protocol


def test_builtin_protocols_have_rules() -> None:
    p = load_protocol("sts_5x")
    assert [r.angle for r in p.rules] == ["trunk_lean", "knee_flexion", "hip_flexion"]
    rules = p.rule_list()
    assert rules[0].segments == ("trunk",)
    assert rules[1].segments == ("thigh", "shank")
    assert rules[0].band.warn == (-10, 55)
    assert load_protocol("sts_30s").rules


def test_rule_defaults() -> None:
    r = RuleSpec(angle="knee_flexion", ok=(0, 100))
    assert r.band().warn == (-10, 110)
    assert r.resolved_segments() == ("thigh", "shank")
    assert r.to_rule().label == "knee_flexion"
    r2 = RuleSpec(angle="trunk_lean", ok=(0, 30), segments=["trunk", "head"], margin=5)
    assert r2.band().warn == (-5, 35) and r2.resolved_segments() == ("trunk", "head")


@pytest.mark.parametrize(
    "kw",
    [
        {"angle": "wing_span", "ok": (0, 10)},
        {"angle": "knee_flexion", "ok": (10, 0)},
        {"angle": "knee_flexion", "ok": (0, 10), "warn": (5, 10)},
        {"angle": "knee_flexion", "ok": (0, 10), "segments": ["tail"]},
        {"angle": "knee_flexion", "ok": (0, 10), "margin": -1},
    ],
)
def test_rule_validation_errors(kw) -> None:
    with pytest.raises(ValidationError):
        RuleSpec(**kw)
