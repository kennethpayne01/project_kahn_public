#!/usr/bin/env python3
"""
API Provider Validation Script
================================
Tests connectivity and JSON response capability for all configured providers.
Run this before starting a game to verify your API keys work.

Usage: python test_providers.py [model_id ...]
  No args: tests all configured providers
  With args: tests only specified models
"""

import sys
import os

# Ensure we can import from current directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_providers import get_llm_response, parse_json_response, list_available_models, check_provider

# Test prompt: requires JSON output similar to the game's Phase 1
TEST_PROMPT = """You are playing a strategic simulation game. Respond with ONLY a JSON object (no markdown, no explanation):

{
    "status": "ready",
    "model_name": "<your model name>",
    "can_reason_strategically": true,
    "test_message": "Hello from Kahn Game test"
}"""

# Recommended models to test
RECOMMENDED_MODELS = {
    "deepseek-chat":      "DeepSeek V3",
    "qwen-max":           "Qwen Max (通义千问)",
    "glm-4-plus":         "GLM-4 Plus (智谱)",
    "moonshot-v1-128k":   "Moonshot Kimi",
}

OPTIONAL_MODELS = {
    "deepseek-reasoner":  "DeepSeek R1 (Reasoner)",
    "qwen-plus":          "Qwen Plus",
    "glm-4-flash":        "GLM-4 Flash",
}


def test_model(model_id: str, label: str) -> bool:
    """Test a single model's connectivity and JSON capability."""
    print(f"\n{'─' * 50}")
    print(f"Testing: {label} ({model_id})")
    print(f"{'─' * 50}")

    # Check API key
    if not check_provider(model_id):
        print(f"  SKIP - API key not configured")
        return False

    try:
        # Call the model
        print(f"  Calling API...", end=" ", flush=True)
        response = get_llm_response(model_id, TEST_PROMPT, temperature=0.3, max_tokens=500, retries=2)
        print(f"OK ({len(response)} chars)")

        # Parse JSON
        print(f"  Parsing JSON...", end=" ", flush=True)
        parsed = parse_json_response(response)

        if parsed and parsed.get("status") == "ready":
            print(f"OK")
            print(f"  Model says: {parsed.get('test_message', '(no message)')}")
            print(f"  PASS")
            return True
        elif parsed:
            print(f"Partial (keys: {list(parsed.keys())})")
            print(f"  WARNING - JSON parsed but missing expected fields")
            print(f"  Raw: {response[:200]}")
            return True  # Still usable
        else:
            print(f"FAILED")
            print(f"  Raw response: {response[:300]}")
            return False

    except Exception as e:
        print(f"ERROR")
        print(f"  {type(e).__name__}: {e}")
        return False


def main():
    print("=" * 60)
    print("Kahn Game - API Provider Validation")
    print("=" * 60)

    # Show configured providers
    available = list_available_models()
    print("\nProvider Status:")
    for provider, entries in available.items():
        for entry in entries:
            status_icon = "OK" if entry["status"] == "configured" else "MISSING"
            print(f"  [{status_icon:>7}] {entry['env_var']:<25} (prefix: {entry['prefix']})")

    # Determine which models to test
    if len(sys.argv) > 1:
        # Test specific models from command line
        models_to_test = {m: m for m in sys.argv[1:]}
    else:
        # Test all recommended models
        models_to_test = {**RECOMMENDED_MODELS, **OPTIONAL_MODELS}

    # Run tests
    results = {}
    for model_id, label in models_to_test.items():
        results[model_id] = test_model(model_id, label)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    passed = [m for m, ok in results.items() if ok]
    skipped = [m for m, ok in results.items() if ok is False and not check_provider(m)]
    failed = [m for m, ok in results.items() if ok is False and check_provider(m)]

    if passed:
        print(f"\n  PASSED ({len(passed)}):")
        for m in passed:
            print(f"    {m}")

    if skipped:
        print(f"\n  SKIPPED - no API key ({len(skipped)}):")
        for m in skipped:
            print(f"    {m}")

    if failed:
        print(f"\n  FAILED ({len(failed)}):")
        for m in failed:
            print(f"    {m}")

    # Game readiness check
    print(f"\n{'─' * 60}")
    if len(passed) >= 2:
        print(f"READY! You have {len(passed)} working models.")
        print(f"Example game command:")
        m1, m2 = passed[0], passed[1]
        print(f"  python kahn_game_open.py --model_a {m1} --model_b {m2} --turns 5 --scenario v6_baseline")
    elif len(passed) == 1:
        print(f"Need at least 2 models. Only {passed[0]} is working.")
        print(f"Configure another API key in .env")
    else:
        print(f"No models available. Please configure API keys in .env")
        print(f"See .env.example for reference.")

    return 0 if len(passed) >= 2 else 1


if __name__ == "__main__":
    sys.exit(main())
