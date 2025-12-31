import os
import runpod
from vllm import AsyncLLMEngine, EngineArgs, SamplingParams
from vllm.utils import random_uuid

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen3-8B"

# 8B Model needs ~16GB VRAM (BF16). 
# A single GPU is sufficient.
TP_SIZE = 1 

# Engine Arguments
# We removed 'enable_reasoning' because it causes crashes in newer vLLM versions.
engine_args = EngineArgs(
    model=MODEL_NAME,
    tensor_parallel_size=TP_SIZE,
    trust_remote_code=True,      # Required for Qwen architectures
    disable_log_stats=False,
    max_model_len=32768,         # Standard context length for Qwen3
    gpu_memory_utilization=0.95, # Use maximum available VRAM
    dtype="auto",                # Automatically use bfloat16 or float16
    enforce_eager=False,         # set to True if you see CUDA graph errors (rare on 8B)
)

print(f"--- Initializing vLLM Engine for {MODEL_NAME} ---")
engine = AsyncLLMEngine.from_engine_args(engine_args)
print("--- Engine Initialized ---")

async def handler(job):
    job_input = job["input"]
    
    # 1. Extract Input
    prompt = job_input.get("prompt")
    messages = job_input.get("messages")
    
    # 2. Handle Chat Format (Simple Template)
    # If the user sends a list of messages, we format it for Qwen
    if messages and not prompt:
        prompt = ""
        for m in messages:
            # Standard Qwen ChatML format
            prompt += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

    if not prompt:
        return {"error": "Missing 'prompt' or 'messages' in input"}

    # 3. Set Sampling Parameters
    # Qwen3 works best with these settings
    sampling_params = SamplingParams(
        temperature=job_input.get("temperature", 0.7),
        top_p=job_input.get("top_p", 0.95),
        max_tokens=job_input.get("max_tokens", 2048),
        stop=["<|im_end|>", "<|endoftext|>"]
    )

    request_id = random_uuid()
    
    # 4. Generate
    results_generator = engine.generate(prompt, sampling_params, request_id)

    # 5. Stream Output
    final_output = ""
    async for request_output in results_generator:
        final_output = request_output.outputs[0].text

    return {"output": final_output}

runpod.serverless.start({"handler": handler})
