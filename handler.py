import os
import runpod
from vllm import AsyncLLMEngine, EngineArgs, SamplingParams
from vllm.utils import random_uuid

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen3-30B-A3B"

# Detect if we are on a large GPU (A100) or need to shard (Multi-GPU)
# Default to 1, but you can change this via Environment Variable in RunPod UI
TP_SIZE = int(os.environ.get("TENSOR_PARALLEL_SIZE", 1))

# Engine Arguments for Qwen3
# Note: enable_reasoning and reasoning_parser are specific to Qwen3's thinking mode
engine_args = EngineArgs(
    model=MODEL_NAME,
    tensor_parallel_size=TP_SIZE,
    trust_remote_code=True,      # Often needed for Qwen architectures
    enable_reasoning=True,       # Enable "Thinking" mode
    reasoning_parser="deepseek_r1", # Qwen3 uses the DeepSeek R1 parser format
    disable_log_stats=False,
    max_model_len=32768,         # Native context length
    gpu_memory_utilization=0.95, # Maximize VRAM usage
)

# Initialize the Engine
print("--- Initializing vLLM Engine ---")
engine = AsyncLLMEngine.from_engine_args(engine_args)
print("--- Engine Initialized ---")

async def handler(job):
    """
    RunPod Handler Function
    """
    job_input = job["input"]
    
    # Extract prompt and parameters
    prompt = job_input.get("prompt")
    messages = job_input.get("messages") # Support for chat format
    
    # If messages are provided, apply chat template (basic implementation)
    if messages and not prompt:
        # You might need a tokenizer here to apply proper chat templates
        # For simplicity, we assume 'prompt' is passed or raw text
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

    if not prompt:
        return {"error": "Missing 'prompt' or 'messages' in input"}

    # specific parameters for generation
    sampling_params = SamplingParams(
        temperature=job_input.get("temperature", 0.6),
        top_p=job_input.get("top_p", 0.95),
        max_tokens=job_input.get("max_tokens", 4096),
    )

    request_id = random_uuid()
    
    # Generate Output
    results_generator = engine.generate(prompt, sampling_params, request_id)

    # Stream or collect result
    final_output = ""
    async for request_output in results_generator:
        final_output = request_output.outputs[0].text

    return {"output": final_output}

# Start the RunPod Serverless Worker
runpod.serverless.start({"handler": handler})
