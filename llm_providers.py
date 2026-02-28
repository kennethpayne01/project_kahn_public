"""
Unified LLM Provider Layer for Kahn Game
=========================================
Supports original models (GPT, Claude, Gemini) and Chinese domestic models
(DeepSeek, Qwen, GLM, Moonshot) via OpenAI-compatible API endpoints.
"""

import os
import json
import time
import logging
import re
from typing import Dict, Any, Optional

from dotenv import load_dotenv

# Load environment variables
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '.env'))

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Provider Configuration
# ─────────────────────────────────────────────

# Model prefix → provider routing
# All Chinese models use OpenAI-compatible endpoints
MODEL_PROVIDERS = {
    # Original models
    "gpt-":      {"provider": "openai",    "env_key": "OPENAI_API_KEY",    "base_url": "https://api.openai.com/v1"},
    "o1-":       {"provider": "openai",    "env_key": "OPENAI_API_KEY",    "base_url": "https://api.openai.com/v1"},
    "o3-":       {"provider": "openai",    "env_key": "OPENAI_API_KEY",    "base_url": "https://api.openai.com/v1"},
    "claude-":   {"provider": "anthropic", "env_key": "ANTHROPIC_API_KEY", "base_url": None},
    "gemini-":   {"provider": "google",    "env_key": "GOOGLE_API_KEY",    "base_url": None},
    # Chinese domestic models (OpenAI-compatible)
    "deepseek-": {"provider": "openai_compat", "env_key": "DEEPSEEK_API_KEY",  "base_url": "https://api.deepseek.com"},
    "qwen-":     {"provider": "openai_compat", "env_key": "DASHSCOPE_API_KEY", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"},
    "glm-":      {"provider": "openai_compat", "env_key": "ZHIPU_API_KEY",     "base_url": "https://open.bigmodel.cn/api/paas/v4"},
    "moonshot-": {"provider": "openai_compat", "env_key": "MOONSHOT_API_KEY",  "base_url": "https://api.moonshot.cn/v1"},
}

# ─────────────────────────────────────────────
# SDK Imports (guarded)
# ─────────────────────────────────────────────

try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

# ─────────────────────────────────────────────
# Client Cache (lazy initialization)
# ─────────────────────────────────────────────

_clients: Dict[str, Any] = {}


def _get_provider_config(model: str) -> Dict[str, Any]:
    """Find provider config by model prefix."""
    m = model.lower()
    for prefix, config in MODEL_PROVIDERS.items():
        if m.startswith(prefix):
            return config
    raise RuntimeError(f"Unsupported model: {model}. Supported prefixes: {list(MODEL_PROVIDERS.keys())}")


def _get_openai_client(api_key: str, base_url: str) -> Any:
    """Get or create an OpenAI-compatible client (cached by base_url)."""
    cache_key = f"openai:{base_url}"
    if cache_key not in _clients:
        if not openai:
            raise RuntimeError("openai package not installed. Run: pip install openai")
        import httpx
        _clients[cache_key] = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.Client(),
        )
    return _clients[cache_key]


def _get_anthropic_client(api_key: str) -> Any:
    """Get or create an Anthropic client."""
    if "anthropic" not in _clients:
        if not anthropic:
            raise RuntimeError("anthropic package not installed. Run: pip install anthropic")
        _clients["anthropic"] = anthropic.Anthropic(api_key=api_key)
    return _clients["anthropic"]


def _get_google_configured(api_key: str):
    """Configure Google GenAI SDK."""
    if "google" not in _clients:
        if not genai:
            raise RuntimeError("google-generativeai package not installed. Run: pip install google-generativeai")
        genai.configure(api_key=api_key)
        _clients["google"] = True


# ─────────────────────────────────────────────
# Main API Function
# ─────────────────────────────────────────────

def get_llm_response(model: str, prompt: str, temperature: float = 0.7,
                     max_tokens: int = 3000, retries: int = 3) -> str:
    """
    Get response from any supported LLM provider.

    Supported models:
        Original: gpt-*, claude-*, gemini-*, o1-*, o3-*
        Chinese:  deepseek-chat, deepseek-reasoner, qwen-max, qwen-plus,
                  glm-4-plus, moonshot-v1-128k, etc.
    """
    config = _get_provider_config(model)
    provider = config["provider"]
    api_key = os.getenv(config["env_key"])

    if not api_key:
        raise RuntimeError(
            f"API key not found for model '{model}'. "
            f"Please set {config['env_key']} in your .env file."
        )

    last_err = None
    for attempt in range(retries):
        try:
            if provider == "openai":
                return _call_openai(model, prompt, temperature, max_tokens, api_key, config["base_url"])
            elif provider == "openai_compat":
                return _call_openai_compat(model, prompt, temperature, max_tokens, api_key, config["base_url"])
            elif provider == "anthropic":
                return _call_anthropic(model, prompt, temperature, max_tokens, api_key)
            elif provider == "google":
                return _call_google(model, prompt, temperature, max_tokens, api_key)
            else:
                raise RuntimeError(f"Unknown provider: {provider}")
        except Exception as e:
            last_err = e
            logger.warning(f"Attempt {attempt + 1}/{retries} failed for {model}: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff

    raise RuntimeError(f"Model call failed after {retries} retries: {last_err}")


# ─────────────────────────────────────────────
# Provider-Specific Implementations
# ─────────────────────────────────────────────

def _call_openai(model: str, prompt: str, temperature: float, max_tokens: int,
                 api_key: str, base_url: str) -> str:
    """Call OpenAI API (GPT, O1, O3 models)."""
    client = _get_openai_client(api_key, base_url)
    m = model.lower()

    # GPT-5.x, o1, o3 models use max_completion_tokens
    is_new_model = 'gpt-5' in m or m.startswith('o1') or m.startswith('o3')
    token_param = 'max_completion_tokens' if is_new_model else 'max_tokens'

    kwargs = {
        'model': model,
        'messages': [{"role": "user", "content": prompt}],
        token_param: max_tokens,
    }
    # Reasoning models (o1/o3) don't support temperature
    if not (m.startswith('o1') or m.startswith('o3')):
        kwargs['temperature'] = temperature

    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content


def _call_openai_compat(model: str, prompt: str, temperature: float, max_tokens: int,
                        api_key: str, base_url: str) -> str:
    """Call OpenAI-compatible API (DeepSeek, Qwen, GLM, Moonshot)."""
    client = _get_openai_client(api_key, base_url)

    kwargs = {
        'model': model,
        'messages': [{"role": "user", "content": prompt}],
        'max_tokens': max_tokens,
        'temperature': temperature,
    }

    # DeepSeek-R1 (reasoner) doesn't support temperature
    if 'reasoner' in model.lower():
        del kwargs['temperature']

    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content


def _call_anthropic(model: str, prompt: str, temperature: float, max_tokens: int,
                    api_key: str) -> str:
    """Call Anthropic API (Claude models)."""
    client = _get_anthropic_client(api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in resp.content)


def _call_google(model: str, prompt: str, temperature: float, max_tokens: int,
                 api_key: str) -> str:
    """Call Google GenAI API (Gemini models)."""
    _get_google_configured(api_key)
    gmodel = genai.GenerativeModel(model)
    resp = gmodel.generate_content(
        [{"text": prompt}],
        generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
    )
    return resp.text


# ─────────────────────────────────────────────
# JSON Parsing (from original code)
# ─────────────────────────────────────────────

def parse_json_response(text: str) -> Dict[str, Any]:
    """Robust JSON parsing with fallback strategies."""
    try:
        result = json.loads(text)
        logger.info(f"JSON keys found: {list(result.keys())}")
        return result
    except Exception:
        pass

    # Try markdown code block
    json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group(1))
            logger.info(f"JSON keys found (from markdown): {list(result.keys())}")
            return result
        except Exception:
            pass

    # Try brace extraction
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(text[start:end + 1])
            logger.info(f"JSON keys found (from extraction): {list(result.keys())}")
            return result
        except Exception:
            pass

    logger.warning(f"No JSON found in text: {text[:200]}...")
    return {}


# ─────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────

def list_available_models() -> Dict[str, list]:
    """List all configured providers and their available API keys."""
    available = {}
    for prefix, config in MODEL_PROVIDERS.items():
        key = os.getenv(config["env_key"])
        status = "configured" if key else "missing"
        provider_name = config["env_key"].replace("_API_KEY", "").replace("_", " ").title()
        if provider_name not in available:
            available[provider_name] = []
        available[provider_name].append({
            "prefix": prefix,
            "status": status,
            "env_var": config["env_key"],
        })
    return available


def check_provider(model: str) -> bool:
    """Check if a model's API key is configured."""
    try:
        config = _get_provider_config(model)
        return bool(os.getenv(config["env_key"]))
    except RuntimeError:
        return False
