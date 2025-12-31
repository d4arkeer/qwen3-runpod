import os
import runpod
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from transformers import AutoTokenizer

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen3-8B"
TP_SIZE = 1 

# --- Load Tokenizer ---
print(f"--- Loading Tokenizer for {MODEL_NAME} ---")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# --- Engine Setup ---
engine_args = AsyncEngineArgs(
    model=MODEL_NAME,
    tensor_parallel_size=TP_SIZE,
    gpu_memory_utilization=0.90,
    max_model_len=32768,         
    dtype="auto",
    disable_log_stats=False,
    trust_remote_code=True,
    worker_use_ray=False,
)

print(f"--- Initializing vLLM for {MODEL_NAME} ---")
engine = AsyncLLMEngine.from_engine_args(engine_args)
print("--- Engine Initialized ---")

async def handler(job):
    job_input = job["input"]
    
    # 1. Extract Inputs
    prompt = job_input.get("prompt")
    messages = job_input.get("messages")
    
    # 2. Dynamic Thinking Switch (Default: False)
    # You can now send "enable_thinking": true in your JSON to activate it.
    should_think = job_input.get("enable_thinking", False)

    # 3. Apply Chat Template
    if messages and not prompt:
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=should_think  # <--- Dynamic Flag
            )
        except Exception as e:
            # Fallback if the specific template version doesn't support the flag yet
            print(f"Template Error: {e}")
            # Retry without the flag (standard generation)
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

    if not prompt:
        return {"error": "Missing 'prompt' or 'messages' in input"}

    # 4. Sampling Parameters
    # Tip: You might want higher tokens if thinking is enabled
    default_max_tokens = 4096 if should_think else 2048
    
    sampling_params = SamplingParams(
        temperature=job_input.get("temperature", 0.7),
        top_p=job_input.get("top_p", 0.95),
        max_tokens=job_input.get("max_tokens", default_max_tokens),
        stop=["<|im_end|>", "<|endoftext|>"]
    )

    request_id = random_uuid()
    results_generator = engine.generate(prompt, sampling_params, request_id)

    final_output = ""
    async for request_output in results_generator:
        final_output = request_output.outputs[0].text

    return {"output": final_output}

runpod.serverless.start({"handler": handler})
