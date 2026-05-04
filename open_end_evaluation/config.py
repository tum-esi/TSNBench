import os
from dotenv import load_dotenv
load_dotenv()

QUESTIONSET_FILE = "input/test_cases"
RESULTS_DIR = "output/CQF"
RUNS_PER_QUESTION = 3

TEMPERATURE_WITH = 0.7      # with temperature
TEMPERATURE_WITHOUT = 0.0      # deterministic

# update it accordingly
MODEL_PRICING = {
    "grok-4-1-fast-reasoning"       : {"input":  0.20, "output": 0.50},
    "grok-4-1-fast-non-reasoning"   : {"input":  0.20, "output": 0.50},
    "o3-2025-04-16"                 : {"input": 2.00, "output": 8.00},
    "gpt-4o-2024-08-06"             : {"input":  2.50, "output": 10.00},
    "gpt-4o-mini-2024-07-18"        : {"input":  0.15, "output":  0.60},
    "gpt-5-2025-08-07"              : {"input":  1.25, "output":  10.00},
    "deepseek-chat"                 : {"input":  0.028, "output":  0.42},
    "deepseek-reasoner"             : {"input":  0.028, "output":  0.42},
    "gemini-2.5-flash"              : {"input":  0.30, "output":  2.50},
    "claude-sonnet-4-5-20250929"    : {"input":  3.00, "output":  15.00},
    "Llama-3.3-70B-Instruct"        : {"input":  0.20, "output":  0.80},
    "mistral-medium-2508"           : {"input":  0.40, "output":  2.00},
    "mistral-large-2512"            : {"input":  0.50, "output":  1.50},
    "llama-3.2-1B"                  : {"input":  0.10, "output":  0.40},
    "Qwen3-8B"                      : {"input":  0.10, "output":  0.30},
    "ministral-8b-2512"             : {"input":  0.15, "output":  0.15}
}

def compute_cost(model_key: str, tokens_in: int, tokens_out: int) -> float:
    """Calculate cost in USD for a single API call."""
    pricing = MODEL_PRICING.get(model_key, {"input": 0.0, "output": 0.0})
    return (tokens_in * pricing["input"] + tokens_out * pricing["output"]) / 1_000_000

from openai import OpenAI
from mistralai import Mistral
from anthropic import Anthropic
from google import genai
from google.genai import types as genai_types

# OpenAI — GPT-4o, GPT-4o mini, o3
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Anthropic — Claude 3.7 Sonnet
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Google — Gemini 2.5 Flash
gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Mistral — Large 3, Medium 3.1
mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

# xAI — Grok 3
grok_client = OpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1"
)

# DeepSeek — R1, V3
deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

# HuggingFace Router — Llama 3.3, Qwen3 8B
hf_client = OpenAI(
    api_key=os.getenv("HF_TOKEN"),
    base_url="https://router.huggingface.co/v1"
)

MODELS = [
    {
        "key"         : "o3-2025-04-16",
        "model_id"    : "o3-2025-04-16",
        "client"      : openai_client,
        "api"         : "openai",
        "pricing_key" : "o3-2025-04-16",
        "type"        : "Reasoning",
        "weights"     : "Closed",
        "org"         : "OpenAI",
        "max_tokens"  : 100000,
        "max_tokens_param": "max_completion_tokens",
    },
    {
        "key"         : "gpt-4o-2024-08-06",
        "model_id"    : "gpt-4o-2024-08-06",
        "client"      : openai_client,
        "api"         : "openai",
        "pricing_key" : "gpt-4o-2024-08-06",
        "type"        : "Standard",
        "weights"     : "Closed",
        "org"         : "OpenAI",
        "max_tokens"  : 16384,
    },
    {
        "key"         : "gpt-4o-mini-2024-07-18",
        "model_id"    : "gpt-4o-mini-2024-07-18",
        "client"      : openai_client,
        "api"         : "openai",
        "pricing_key" : "gpt-4o-mini-2024-07-18",
        "type"        : "Standard",
        "weights"     : "Closed",
        "org"         : "OpenAI",
        "max_tokens"  : 16384,
    },
    {
        "key"         : "gpt-5-2025-08-07",
        "model_id"    : "gpt-5-2025-08-07",
        "client"      : openai_client,
        "api"         : "openai",
        "pricing_key" : "gpt-5-2025-08-07",
        "type"        : "Standard",
        "weights"     : "Closed",
        "org"         : "OpenAI",
        "max_tokens"  : 32768,
        "max_tokens_param": "max_completion_tokens",
        "skip_temperature" : True,
    },
    {
        "key"         : "claude-sonnet-4-5-20250929",
        "model_id"    : "claude-sonnet-4-5-20250929",
        "client"      : anthropic_client,
        "api"         : "anthropic",
        "pricing_key" : "claude-sonnet-4-5-20250929",
        "type"        : "Standard",
        "weights"     : "Closed",
        "org"         : "Anthropic",
        "max_tokens"  : 64000,
    },
    {
        "key"         : "gemini-2.5-flash",
        "model_id"    : "gemini-2.5-flash",
        "client"      : gemini_client,
        "api"         : "gemini",
        "pricing_key" : "gemini-2.5-flash",
        "type"        : "Standard",
        "weights"     : "Closed",
        "org"         : "Google",
        "max_tokens"  : 65536,
    },
    {
        "key"         : "grok-4-1-fast-reasoning",
        "model_id"    : "grok-4-1-fast-reasoning",
        "client"      : grok_client,
        "api"         : "openai",
        "pricing_key" : "grok-4-1-fast-reasoning",
        "type"        : "Reasoning",
        "weights"     : "Closed",
        "org"         : "xAI",
        "max_tokens"  : 131072,
    },
    {
        "key"         : "grok-4-1-fast-non-reasoning",
        "model_id"    : "grok-4-1-fast-non-reasoning",
        "client"      : grok_client,
        "api"         : "openai",
        "pricing_key" : "grok-4-1-fast-non-reasoning",
        "type"        : "Standard",
        "weights"     : "Closed",
        "org"         : "xAI",
        "max_tokens"  : 131072,
    },
    {
        "key"         : "deepseek-chat",
        "model_id"    : "deepseek-chat",
        "client"      : deepseek_client,
        "api"         : "openai",
        "pricing_key" : "deepseek-chat",
        "type"        : "Standard",
        "weights"     : "Open",
        "org"         : "DeepSeek",
        "max_tokens"  : 8192,
    },
    {
        "key"         : "deepseek-reasoner",
        "model_id"    : "deepseek-reasoner",
        "client"      : deepseek_client,
        "api"         : "openai",
        "pricing_key" : "deepseek-reasoner",
        "type"        : "Reasoning",
        "weights"     : "Open",
        "org"         : "DeepSeek",
        "max_tokens"  : 8000,
    },
    {
        "key"         : "mistral-large-2512",
        "model_id"    : "mistral-large-2512",
        "client"      : mistral_client,
        "api"         : "mistral",
        "pricing_key" : "mistral-large-2512",
        "type"        : "Standard",
        "weights"     : "Partial",
        "org"         : "Mistral AI",
        "max_tokens"  : 32768,
    },
    {
        "key"         : "mistral-medium-2508",
        "model_id"    : "mistral-medium-2508",
        "client"      : mistral_client,
        "api"         : "mistral",
        "pricing_key" : "mistral-medium-2508",
        "type"        : "Standard",
        "weights"     : "Partial",
        "org"         : "Mistral AI",
        "max_tokens"  : 32768,
    },
    {
        "key"         : "ministral-8b-2512",
        "model_id"    : "ministral-8b-2512",
        "client"      : mistral_client,
        "api"         : "mistral",
        "pricing_key" : "ministral-8b-2512",
        "type"        : "Standard",
        "weights"     : "Partial",
        "org"         : "Mistral AI",
        "max_tokens"  : 32768,
    },
    {
        "key"         : "Llama-3.3-70B-Instruct",
        "model_id"    : "meta-llama/Llama-3.3-70B-Instruct:groq",
        "client"      : hf_client,
        "api"         : "openai",
        "pricing_key" : "Llama-3.3-70B-Instruct",
        "type"        : "Standard",
        "weights"     : "Open",
        "org"         : "Meta",
        "max_tokens"  : 8192,
    },
    {
        "key"         : "Llama-3.2-1B",
        "model_id"    : "meta-llama/Llama-3.2-1B-Instruct:novita",
        "client"      : hf_client,
        "api"         : "openai",
        "pricing_key" : "Llama-3.2-1B",
        "type"        : "Standard",
        "weights"     : "Open",
        "org"         : "Meta",
        "max_tokens"  : 8192,
    },
    {
        "key"         : "Qwen3-8B",
        "model_id"    : "Qwen/Qwen3-8B:nscale",
        "client"      : hf_client,
        "api"         : "openai",
        "pricing_key" : "Qwen3-8B",
        "type"        : "Standard",
        "weights"     : "Open",
        "org"         : "Alibaba",
        "max_tokens"  : 32768,
    },
]