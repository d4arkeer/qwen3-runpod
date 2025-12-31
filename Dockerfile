# Use the latest vLLM image
FROM vllm/vllm-openai:latest

# Install RunPod SDK
RUN pip install runpod

# Copy your handler script
COPY handler.py /handler.py

# --- CRITICAL FIX ---
# We must reset the entrypoint so Docker doesn't force the default vLLM server command.
ENTRYPOINT []

# Run your handler
CMD ["python3", "-u", "/handler.py"]
