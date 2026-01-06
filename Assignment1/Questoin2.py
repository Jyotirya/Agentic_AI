from transformers import pipeline

# Create a text generation pipeline
generator = pipeline("text-generation")

# Define the prompt
prompt = "In a distant future, AI will take over human jobs"

# Generate text with two different continuations
outputs = generator(
    prompt,
    num_return_sequences=2,
    max_new_tokens=50,
    do_sample=True
)

# Print the generated continuations clearly separated
for i, output in enumerate(outputs, start=1):
    print(f"\n--- Generated Continuation {i} ---")
    print(output["generated_text"])