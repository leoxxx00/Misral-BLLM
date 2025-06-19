from llama_cpp import Llama

model_path = "Mistral-7B-Instruct-v0.1.Q2_K.gguf"
llm = Llama(model_path=model_path)

print("Welcome to the terminal chat! Type 'exit' to quit.\n")

chat_history = []

while True:
    user_input = input("You: ")
    if user_input.strip().lower() == "exit":
        print("Goodbye!")
        break

    chat_history.append({"role": "user", "content": user_input})

    response = llm.create_chat_completion(messages=chat_history)
    # Access response as dict:
    reply = response['choices'][0]['message']['content'].strip()

    print("Bot:", reply)

    chat_history.append({"role": "assistant", "content": reply})
