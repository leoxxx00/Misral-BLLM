from llama_cpp import Llama

# Load your GGUF quantized model
model_path = "Mistral-7B-Instruct-v0.1.Q2_K.gguf"

# Initialize the Llama model with chat format and context length
llm = Llama(
    model_path=model_path,
    chat_format="chatml",     # or "llama-2" depending on your model's format
    n_ctx=4096,               # context window size
    temperature=0.7,          # controls randomness (0 = deterministic, 1 = very random)
    top_p=0.9,                # nucleus sampling (probability mass to consider)
    top_k=40,                 # limits sampling to top-k tokens
    repeat_penalty=1.1,       # discourages repetition
    verbose=False             # toggle detailed backend logs
)

print("Welcome to the terminal chat! Type 'exit' to quit.\n")

# Start with a system message to guide behavior
chat_history = [
    {"role": "system", "content": "You are a helpful, concise, and intelligent AI assistant."}
]

# Interactive loop
while True:
    user_input = input("You: ")
    if user_input.strip().lower() == "exit":
        print("Goodbye!")
        break

    chat_history.append({"role": "user", "content": user_input})

    # Generate a response from the assistant
    response = llm.create_chat_completion(messages=chat_history)

    # Extract content from the response
    reply = response['choices'][0]['message']['content'].strip()
    print("Bot:", reply)

    # Add the assistant's reply to history
    chat_history.append({"role": "assistant", "content": reply})
