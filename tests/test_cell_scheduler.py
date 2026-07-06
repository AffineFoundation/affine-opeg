"""Focused unit tests for the weighted cell scheduler pure logic."""
from affine_opeg.application.cell_scheduler import (
    parse_pool_config, _env_targets, DEFAULT_CELL_TTL_S,
)


def _base(pool_extra=None, top_extra=None):
    pool = {
        "envs": ["a", "a", "a", "b", "c"],  # multiplicity: a=3 b=1 c=1
        "teachers": ["t1", "t2"],
        "task_id_pool": {"a": [0, 1, 2], "b": [0, 1], "c": [0]},
    }
    if pool_extra:
        pool.update(pool_extra)
    cfg = {"pool": pool, "target_active_cells": 100, "target_samples": 12}
    if top_extra:
        cfg.update(top_extra)
    return cfg


def test_backcompat_multiplicity_weights():
    p = parse_pool_config(_base())
    assert p.envs == ("a", "b", "c")  # distinct, order preserved
    assert p.env_weights == {"a": 3.0, "b": 1.0, "c": 1.0}
    assert p.cell_ttl_s == DEFAULT_CELL_TTL_S
    t = _env_targets(p)
    # a:3/5*100=60, b:1/5*100=20, c:20
    assert t == {"a": 60, "b": 20, "c": 20}


def test_explicit_weights_override():
    p = parse_pool_config(_base(pool_extra={"env_weights": {"a": 8, "b": 2, "c": 0}}))
    assert p.env_weights == {"a": 8.0, "b": 2.0, "c": 0.0}
    t = _env_targets(p)
    # a:8/10*100=80, b:20, c:floored to min_active_per_env=1 (weight 0 -> probe)
    assert t["a"] == 80 and t["b"] == 20
    assert t["c"] == 1  # cut env keeps a cold-start recovery probe


def test_min_active_floor_configurable():
    p = parse_pool_config(_base(
        pool_extra={"env_weights": {"a": 100, "b": 0, "c": 0}},
        top_extra={"min_active_per_env": 5},
    ))
    t = _env_targets(p)
    assert t["b"] == 5 and t["c"] == 5  # every listed env keeps >=floor
    assert t["a"] == 100  # ~100/100*100


def test_ttl_override():
    p = parse_pool_config(_base(top_extra={"cell_ttl_s": 3600}))
    assert p.cell_ttl_s == 3600


# --- reweight controller pure logic ---
from affine_opeg.publishing.reweighter import compute_weights  # noqa: E402


def test_reweight_cut_warmup_aggressive():
    envs = ["hot", "mid", "dud", "fresh"]
    stats = {
        "hot": (0.50, 200),    # high yield, mature
        "mid": (0.20, 200),    # ok yield, mature
        "dud": (0.03, 200),    # below cut, mature -> CUT
        "fresh": (0.00, 5),    # too few cells -> warmup, NOT cut
    }
    w, lab = compute_weights(envs, stats, cut_yield=0.10, alpha=2.0, min_cells=30, floor=0.03)
    assert lab["dud"].startswith("CUT")
    assert lab["fresh"].startswith("warmup")
    # dud gets only the floor; hot >> mid; fresh (warmup) not floored to 0
    assert w["dud"] == 0.03
    assert w["hot"] > w["mid"] > w["dud"]
    assert w["fresh"] > w["dud"]  # cold-start env keeps a real (non-cut) share
    # aggressive: hot score 0.25 vs mid 0.04 -> >5x ratio above the shared floor
    assert (w["hot"] - 0.03) / (w["mid"] - 0.03) > 5


def test_reweight_no_data_uniform():
    envs = ["a", "b"]
    stats = {}  # nothing sampled yet
    w, lab = compute_weights(envs, stats, cut_yield=0.10, alpha=2.0, min_cells=30, floor=0.03)
    assert w["a"] == w["b"] and all(v > 0 for v in w.values())
    assert all(l.startswith("warmup") for l in lab.values())


def test_reweight_pin_floor():
    # swe has low yield -> would be CUT to the floor, but a pin raises it.
    envs = ["hot", "swe-rebench"]
    stats = {"hot": (0.50, 200), "swe-rebench": (0.06, 200)}
    w, lab = compute_weights(
        envs, stats, cut_yield=0.10, alpha=2.0, min_cells=30, floor=0.03,
        pins={"swe-rebench": 0.10},
    )
    assert w["swe-rebench"] == 0.10  # pinned above its floor(0.03)
    assert "pinned" in lab["swe-rebench"] and "CUT" in lab["swe-rebench"]
    # pin gives swe a real share (~10/(10+28)) rather than the ~3% floor
    total = sum(w.values())
    assert 0.20 < w["swe-rebench"] / total < 0.30


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
    print("ALL PASS")
