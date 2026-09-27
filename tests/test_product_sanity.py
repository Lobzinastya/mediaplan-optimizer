from scripts.run_product_sanity import run_product_sanity


def test_full_product_workflow_sanity():
    result = run_product_sanity()
    assert result["status"] == "passed"
    assert result["days_completed"] == 21
    assert result["history_preserved"] is True
    assert result["duplicate_day_channels"] is False
