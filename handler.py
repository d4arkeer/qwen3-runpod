import os
import runpod
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen3-8B"
TP_SIZE = 1  # 8B fits on 1 GPU

# --- CRITICAL FIX ---
# Use AsyncEngineArgs instead of EngineArgs.
# The V1 engine requires async-specific arguments (like enable_log_requests)
# that are missing from the base EngineArgs class.
engine_args = AsyncEngineArgs(
    model=MODEL_NAME,
    tensor_parallel_size=TP_SIZE,
    gpu_memory_utilization=0.95, 
    max_model_len=32768,         
    dtype="auto",
    disable_log_stats=False,
    # Qwen3 is natively supported in vLLM now, so we don't strictly need 
    # trust_remote_code=True, but we can leave it if specific custom layers are needed.
    # The error regarding 'enable_log_requests' is solved by using AsyncEngineArgs.
)

print(f"--- Initializing vLLM Engine for {MODEL_NAME} ---")
engine = AsyncLLMEngine.from_engine_args(engine_args)
print("--- Engine Initialized ---")

async def handler(job):
    job_input = job["input"]
    prompt = job_input.get("prompt")
    messages = job_input.get("messages")
    
    # Handle Chat Format
    if messages and not prompt:
        prompt = ""
        for m in messages:
            prompt += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

    if not prompt:
        return {"error": "Missing 'prompt' or 'messages' in input"}

    # Sampling Parameters
    sampling_params = SamplingParams(
        temperature=job_input.get("temperature", 0.7),
        top_p=job_input.get("top_p", 0.95),
        max_tokens=job_input.get("max_tokens", 2048),
        stop=["<|im_end|>", "<|endoftext|>"]
    )

    request_id = random_uuid()
    
    # Generate
    results_generator = engine.generate(prompt, sampling_params, request_id)

    # Stream Output
    final_output = ""
    async for request_output in results_generator:
        final_output = request_output.outputs[0].text

    return {"output": final_output}

runpod.serverless.start({"handler": handler})
