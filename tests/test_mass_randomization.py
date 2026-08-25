from scripts.randomization_audit import audit


def test_full_randomization_with_10000_models():
    report = audit(stage="full", samples=10_000, seed=20260824)
    assert report["all_checks_passed"] is True
