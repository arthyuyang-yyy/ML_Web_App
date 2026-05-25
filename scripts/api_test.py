from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import app


def main() -> None:
    client = TestClient(app)

    health = client.get("/health")
    health.raise_for_status()

    response = client.post(
        "/predict",
        json={"seconds": 50, "latitude": 44, "longitude": -12},
    )
    response.raise_for_status()

    result = response.json()
    assert "country" in result
    assert "country_code" in result

    print(f"API prediction test passed: {result}")


if __name__ == "__main__":
    main()

