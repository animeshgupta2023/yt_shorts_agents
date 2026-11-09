from llama_cpp import Llama
model_path = "./models/Qwen3.5-4B-Q4_K_M.gguf"

llm = Llama(
    model_path=str(model_path),
    n_ctx=4066,
    n_gpu_layers=-1,
    verbose=False,
)

while True:
    user_input = input("Enter your prompt (or 'exit' to quit): ")
    if user_input.lower() == "exit":
        break

    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "you are a helpful assistant."},
            {"role": "user", "content": user_input},
        ],
        temperature=0.2,
        max_tokens=1500,
    )

    print("Response:", response["choices"][0]["message"]["content"].strip())