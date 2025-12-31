# Use the latest vLLM image
FROM vllm/vllm-openai:latest

# Install RunPod SDK
RUN pip install runpod

# Copy your handler script
COPY handler.py /handler.py

# --- CRITICAL FIX ---
# Reset the entrypoint to avoid conflict with the base image's default command
ENTRYPOINT []

# Set the command to run your handler
CMD ["python3", "-u", "/handler.py"]
