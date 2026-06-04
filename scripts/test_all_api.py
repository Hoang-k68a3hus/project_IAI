"""End-to-end smoke tests for the backend API.

This script is intentionally dependency-free so it can run both inside the
Docker API container and from a local Python environment.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[1]))


def request_json(
    method: str,
    path: str,
    payload: Optional[Dict[str, Any]] = None,
    timeout: int = 60,
) -> Tuple[int, Any]:
    body = None
    headers = {"Accept": "application/json"}

    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = Request(f"{BASE_URL}{path}", data=body, headers=headers, method=method)

    with urlopen(req, timeout=timeout) as response:
        data = response.read()
        if not data:
            return response.status, None
        return response.status, json.loads(data.decode("utf-8"))


def wait_for_api(timeout_seconds: int = 300) -> Dict[str, Any]:
    deadline = time.time() + timeout_seconds
    last_error = ""

    while time.time() < deadline:
        try:
            status, data = request_json("GET", "/health", timeout=5)
            if status == 200:
                return data
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
            time.sleep(5)

    raise RuntimeError(f"API did not become ready: {last_error}")


def load_sample_ids() -> Tuple[int, int, int]:
    processed_dir = PROJECT_DIR / "data" / "processed"
    trainable_user_id = 14
    product_id = 0

    trainable_path = processed_dir / "trainable_user_mapping.json"
    if trainable_path.exists():
        mapping = json.loads(trainable_path.read_text(encoding="utf-8"))
        cf_to_u = mapping.get("u_idx_cf_to_u_idx") or {}
        if cf_to_u:
            first_key = sorted(cf_to_u, key=lambda x: int(x))[0]
            trainable_user_id = int(cf_to_u[first_key])

    mappings_path = processed_dir / "user_item_mappings.json"
    cold_user_id = 999999999
    if mappings_path.exists():
        mappings = json.loads(mappings_path.read_text(encoding="utf-8"))
        metadata = mappings.get("metadata", {})
        cold_user_id = int(metadata.get("num_users", 0)) + 1_000_000

        idx_to_item = mappings.get("idx_to_item") or {}
        if idx_to_item:
            first_key = sorted(idx_to_item, key=lambda x: int(x))[0]
            product_id = int(idx_to_item[first_key])

    return trainable_user_id, cold_user_id, product_id


def compact(value: Any, limit: int = 180) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return text[:limit] + ("..." if len(text) > limit else "")


def run_check(
    name: str,
    method: str,
    path: str,
    payload: Optional[Dict[str, Any]],
    validator: Callable[[Any], bool],
    timeout: int = 60,
) -> Dict[str, Any]:
    try:
        status, data = request_json(method, path, payload, timeout=timeout)
        ok = status == 200 and validator(data)
        return {"name": name, "ok": ok, "status": status, "detail": compact(data)}
    except HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8")
        except Exception:
            detail = str(exc)
        return {"name": name, "ok": False, "status": exc.code, "detail": detail}
    except Exception as exc:
        return {
            "name": name,
            "ok": False,
            "status": "error",
            "detail": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    health = wait_for_api()
    trainable_user_id, cold_user_id, product_id = load_sample_ids()

    checks = [
        (
            "health",
            "GET",
            "/health",
            None,
            lambda r: r.get("status") == "healthy"
            and not r.get("empty_mode")
            and r.get("num_users", 0) > 0
            and r.get("num_items", 0) > 0,
            60,
        ),
        (
            "model_info",
            "GET",
            "/model_info",
            None,
            lambda r: bool(r.get("model_id"))
            and r.get("num_users", 0) > 0
            and r.get("num_items", 0) > 0,
            60,
        ),
        (
            "stats",
            "GET",
            "/stats",
            None,
            lambda r: r.get("total_users", 0) > 0 and r.get("num_items", 0) > 0,
            60,
        ),
        ("cache_stats", "GET", "/cache_stats", None, lambda r: "caches" in r, 60),
        (
            "recommend_cf_user",
            "POST",
            "/recommend",
            {"user_id": trainable_user_id, "topk": 5, "exclude_seen": True},
            lambda r: r.get("count", 0) > 0
            and len(r.get("recommendations", [])) == r.get("count")
            and not r.get("is_fallback"),
            120,
        ),
        (
            "recommend_cold_user",
            "POST",
            "/recommend",
            {"user_id": cold_user_id, "topk": 5, "exclude_seen": True},
            lambda r: r.get("count", 0) > 0 and r.get("is_fallback") is True,
            120,
        ),
        (
            "batch_recommend",
            "POST",
            "/batch_recommend",
            {
                "user_ids": [trainable_user_id, trainable_user_id + 15, cold_user_id],
                "topk": 3,
                "exclude_seen": True,
            },
            lambda r: r.get("num_users") == 3
            and r.get("fallback_users", 0) >= 1
            and len(r.get("results", {})) == 3,
            120,
        ),
        (
            "similar_items_cf",
            "POST",
            "/similar_items",
            {"product_id": product_id, "topk": 5, "use_cf": True},
            lambda r: r.get("count", 0) > 0
            and len(r.get("similar_items", [])) == r.get("count"),
            120,
        ),
        (
            "similar_items_phobert",
            "POST",
            "/similar_items",
            {"product_id": product_id, "topk": 5, "use_cf": False},
            lambda r: r.get("count", 0) > 0
            and len(r.get("similar_items", [])) == r.get("count"),
            120,
        ),
        ("search_filters", "GET", "/search/filters", None, lambda r: bool(r), 120),
        ("search_stats", "GET", "/search/stats", None, lambda r: bool(r), 120),
        (
            "search",
            "POST",
            "/search",
            {"query": "kem duong da", "topk": 5, "rerank": True},
            lambda r: r.get("count", 0) > 0 and len(r.get("results", [])) == r.get("count"),
            180,
        ),
        (
            "search_similar",
            "POST",
            "/search/similar",
            {"product_id": product_id, "topk": 5, "exclude_self": True},
            lambda r: r.get("count", 0) > 0 and len(r.get("results", [])) == r.get("count"),
            180,
        ),
        (
            "search_profile",
            "POST",
            "/search/profile",
            {
                "product_history": [product_id, product_id + 1],
                "topk": 5,
                "exclude_history": True,
            },
            lambda r: r.get("count", 0) > 0 and len(r.get("results", [])) == r.get("count"),
            180,
        ),
        (
            "evaluate_metrics",
            "POST",
            "/evaluate/metrics",
            {
                "predictions": [[1, 2, 3], [2, 3, 4]],
                "ground_truth": [[2, 5], [4]],
                "metric": "recall",
                "k": 3,
            },
            lambda r: r.get("metric") == "recall" and r.get("value", -1) >= 0,
            60,
        ),
        (
            "reload_model",
            "POST",
            "/reload_model",
            {},
            lambda r: r.get("status") in {"reloaded", "no_update"},
            120,
        ),
    ]

    print("API base:", BASE_URL)
    print("Model:", health.get("model_id"))
    print(
        "Sample ids:",
        f"user={trainable_user_id}",
        f"cold={cold_user_id}",
        f"product={product_id}",
    )
    print()

    results = [
        run_check(name, method, path, payload, validator, timeout)
        for name, method, path, payload, validator, timeout in checks
    ]

    name_width = max(len(r["name"]) for r in results)
    for result in results:
        marker = "PASS" if result["ok"] else "FAIL"
        print(
            f"{marker:4} {result['name']:<{name_width}} "
            f"status={result['status']} detail={result['detail']}"
        )

    failed = [r for r in results if not r["ok"]]
    print()
    print(f"Result: {len(results) - len(failed)}/{len(results)} passed")

    if failed:
        print("Failed checks:", ", ".join(r["name"] for r in failed))
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
