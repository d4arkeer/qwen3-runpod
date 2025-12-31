# Use the latest vLLM image to ensure Qwen3 support
FROM vllm/vllm-openai:latest

# Install RunPod SDK
RUN pip install runpod

# Copy your handler script
COPY handler.py /handler.py

# Set the entrypoint to run your handler
CMD ["python3", "-u", "/handler.py"]
