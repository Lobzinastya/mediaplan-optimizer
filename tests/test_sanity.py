from scripts.run_sanity import run_sanity_checks


def test_numerical_sanity_suite():
    result = run_sanity_checks()
    assert result["status"] == "passed"
