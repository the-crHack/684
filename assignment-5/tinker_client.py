"""Small cached Tinker adapter for this assignment; no API calls on import."""
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent / ".hf-cache"))
import tinker

MODEL = "Qwen/Qwen3-8B"  # Default model; pass model=... to generate() for others.
ROOT = Path(__file__).resolve().parent
_service = None
_clients = {}  # model name -> (sampling client, tokenizer)
paid_calls = 0


async def connect(model=MODEL):
    """Create (once per model) a sampling client and return (sampler, tokenizer)."""
    global _service
    if model not in _clients:
        if _service is None:
            _service = tinker.ServiceClient()
        sampler = await _service.create_sampling_client(base_model=model)
        _clients[model] = (sampler, sampler.get_tokenizer())
    return _clients[model]


async def generate(messages, *, model=MODEL, allow_paid=False, max_paid_calls=10, seed=684,
                   max_tokens=256, temperature=0.0, input_rate=None, output_rate=None):
    """Cache exact requests. Rates are USD/million tokens; unknown costs stay null.

    Call connect() first before concurrent requests. The cap is per Python process,
    not an account spending limit.
    SDK retries can still affect actual billing; reconcile with the dashboard.
    enable_thinking=False is passed to every chat template; templates without a
    thinking mode (e.g. Llama) ignore it.
    """
    global paid_calls
    if not allow_paid:
        raise RuntimeError("Paid calls disabled. Authenticate, then enable the selected notebook run.")
    sampler, tokenizer = await connect(model)
    tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                          enable_thinking=False, tokenize=True,
                                          return_dict=False)
    request = dict(model=model, messages=messages, prompt_tokens=tokens,
                   seed=seed, max_tokens=max_tokens, temperature=temperature,
                   enable_thinking=False, stop=[tokenizer.eos_token_id])
    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    cache_dir = ROOT / "cache"
    cache_dir.mkdir(exist_ok=True)
    path = cache_dir / f"{key}.json"
    if path.exists():
        return dict(json.loads(path.read_text(encoding="utf-8")), cache_hit=True)
    if paid_calls >= max_paid_calls:
        raise RuntimeError(f"Reached the per-kernel cap of {max_paid_calls} new requests.")
    paid_calls += 1
    start = time.perf_counter()
    result = await sampler.sample(
        prompt=tinker.types.ModelInput.from_ints(tokens), num_samples=1,
        sampling_params=tinker.types.SamplingParams(
            max_tokens=max_tokens, temperature=temperature, seed=seed,
            stop=[tokenizer.eos_token_id]))
    sequence = result.sequences[0]
    estimate = None if input_rate is None or output_rate is None else (
        len(tokens) * input_rate + len(sequence.tokens) * output_rate) / 1_000_000
    record = dict(request_id=key, request=request,
                  text=tokenizer.decode(sequence.tokens, skip_special_tokens=True).strip(),
                  output_tokens=list(sequence.tokens), input_token_count=len(tokens),
                  output_token_count=len(sequence.tokens), stop_reason=sequence.stop_reason,
                  elapsed_seconds=time.perf_counter() - start,
                  estimated_uncached_usd=estimate,
                  rates_usd_per_million=dict(input=input_rate, output=output_rate))
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return dict(record, cache_hit=False)
