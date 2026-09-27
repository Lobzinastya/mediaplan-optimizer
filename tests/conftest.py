from __future__ import annotations

import pytest

from mediaplan_optimizer.campaign import start_campaign
from mediaplan_optimizer.catalog import generate_catalog
from mediaplan_optimizer.schemas import OptimizationRequest
from mediaplan_optimizer.service import optimize_media_plan


@pytest.fixture(scope="session")
def product_catalog():
    return generate_catalog(seed=42)


@pytest.fixture(scope="session")
def product_request():
    return OptimizationRequest(
        task_type="A",
        horizon_days=21,
        objective_metric="conversions",
        budget=1_200_000,
    )


@pytest.fixture(scope="session")
def product_plan(product_catalog, product_request):
    return optimize_media_plan(product_request, product_catalog)


@pytest.fixture(scope="session")
def product_state(product_catalog, product_request, product_plan):
    return start_campaign(
        "Product test",
        product_request,
        product_plan,
        product_catalog,
        campaign_id="product-test",
    )
