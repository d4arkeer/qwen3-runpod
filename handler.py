import os
import runpod
from vllm import AsyncLLMEngine, EngineArgs, SamplingParams
from vllm.utils import random_uuid

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen3-30B-A3B"

# Qwen3-30B-A3B is a MoE model.
# It has 30B total parameters (requiring ~60GB VRAM to load)
# But only ~3B active parameters (running very fast).
# You need an A100 80GB or 2x A6000 (with TP=2).

TP_SIZE = int(os.environ.get("TENSOR_PARALLEL_SIZE", 1))

# Engine Arguments (FIXED)
engine_args = EngineArgs(
    model=MODEL_NAME,
    tensor_parallel_size=TP_SIZE,
    trust_remote_code=True,      # Required for Qwen3 architecture
    disable_log_stats=False,
    max_model_len=32768,         # Native context for Qwen3
    gpu_memory_utilization=0.95, # Maximize VRAM usage
    dtype="auto"                 # Auto-detect BF16/FP16
)

# Initialize the Engine
print(f"--- Initializing vLLM Engine for {MODEL_NAME} ---")
engine = AsyncLLMEngine.from_engine_args(engine_args)
print("--- Engine Initialized ---")

async def handler(job):
    job_input = job["input"]
    
    # Extract prompt
    prompt = job_input.get("prompt")
    messages = job_input.get("messages") 
    
    # Basic Chat Template handling
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
        max_tokens=job_input.get("max_tokens", 4096),
        stop=["<|im_end|>", "<|endoftext|>"] # Stop tokens for Qwen
    )

    request_id = random_uuid()
    
    # Generate
    results_generator = engine.generate(prompt, sampling_params, request_id)

    # Stream output
    final_output = ""
    async for request_output in results_generator:
        final_output = request_output.outputs[0].text

    return {"output": final_output}

runpod.serverless.start({"handler": handler})
