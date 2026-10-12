"""Synthetic population generator: determinism, calibration, and drift mode."""

from readmit.fhir.synthetic import PopulationGenerator, write_population


def test_deterministic(tmp_path):
    write_population(tmp_path / "a", 20, seed=9)
    write_population(tmp_path / "b", 20, seed=9)
    for name in ("Patient.ndjson", "Encounter.ndjson", "Observation.ndjson"):
        assert (tmp_path / "a" / name).read_text() == (tmp_path / "b" / name).read_text()


def test_readmission_rate_is_plausible():
    generator = PopulationGenerator(seed=21)
    generator.generate(600)
    rate = generator.stats["readmissions"] / generator.stats["acute_admissions"]
    assert 0.08 < rate < 0.25  # US all-cause 30-day readmission is roughly 14-15%


def test_drift_mode_increases_utilization():
    normal, drifted = PopulationGenerator(seed=4), PopulationGenerator(seed=4, drift=True)
    count = lambda g: sum(  # noqa: E731
        r["resourceType"] == "Encounter" and r["class"]["code"] == "EMER"
        for patient in g.generate(200)
        for r in patient
    )
    assert count(drifted) > count(normal) * 1.2
