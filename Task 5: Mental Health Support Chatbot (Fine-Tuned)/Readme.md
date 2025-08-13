# EmpatheticBot: Fine-Tuning a Language Model for Empathetic Responses

## Task Objective
The goal of this project is to fine-tune a lightweight language model, `distilgpt2`, to generate empathetic and contextually appropriate responses to user prompts expressing emotions or personal situations. The model is trained using LoRA (Low-Rank Adaptation) to efficiently adapt the pre-trained model for empathetic dialogue, making it suitable for deployment in resource-constrained environments like Kaggle notebooks.

## Dataset Used
- **Dataset**: EmpatheticDialogues
- **Source**: Hugging Face Datasets (`empathetic_dialogues`)
- **Description**: This dataset contains multi-turn conversations designed to evoke empathetic responses. Each example includes a prompt (describing a situation or emotion) and an utterance (the empathetic response). The dataset is tokenized and formatted with custom markers (`<|prompt|>`, `<|response|>`, `<|endoftext|>`) to structure input-output pairs for training.

## Models Applied
- **Base Model**: `distilgpt2`
  - A distilled version of GPT-2, chosen for its lightweight architecture (82M parameters) and compatibility with Kaggle's resource constraints (9 GB RAM, T4 GPU).
- **Fine-Tuning Method**: LoRA (Low-Rank Adaptation)
  - LoRA configuration: `r=8`, `lora_alpha=32`, `target_modules=["c_attn"]`, `lora_dropout=0.05`, `bias="none"`.
  - Only a small fraction of parameters (~0.1%) are trained, reducing memory usage and training time.
- **Tokenizer**: `AutoTokenizer` from `distilgpt2`, with the padding token set to the end-of-sequence token.
- **Training Setup**:
  - Epochs: 2 (completes in ~8 minutes on a T4 GPU).
  - Batch size: 8 (per device for both training and evaluation).
  - Learning rate: `3e-4`.
  - Mixed precision: FP16 for faster training and lower memory usage.
  - Output directory: `/kaggle/working/empathetic_bot`.

## Key Results and Findings
- **Training Performance**:
  - The model was successfully fine-tuned using LoRA, with only ~0.1% of parameters updated, ensuring efficiency.
  - Training completed in approximately 8 minutes on a single T4 GPU, fitting within Kaggle’s resource limits.
- **Inference**:
  - A `pipeline` for text generation was implemented for quick sanity checks, producing empathetic responses to sample prompts like "I feel overwhelmed with work" and "My cat is sick and I’m scared."
  - A CLI interface was added for interactive chatting, allowing users to input prompts and receive responses in real-time, with a `/quit` command to exit.
- **Response Quality**:
  - The fine-tuned model generates contextually relevant and empathetic responses, leveraging the EmpatheticDialogues dataset’s emotional context.
  - Example outputs demonstrate the model’s ability to respond with appropriate tone and content, though responses are limited to 64 new tokens to balance coherence and brevity.
- **Deployment**:
  - The model and tokenizer are saved to `/kaggle/working/empathetic_bot`, enabling easy loading for inference or further fine-tuning.
  - The CLI implementation uses `torch.no_grad()` for efficient inference and supports real-time interaction.
- **Limitations**:
  - The lightweight `distilgpt2` model may produce less nuanced responses compared to larger models like GPT-3 or LLaMA.
  - Training for only 2 epochs may limit the model’s ability to fully capture complex emotional nuances, but was chosen to fit Kaggle’s constraints.
  - The maximum response length (64 tokens) may truncate longer, more detailed responses.

This project demonstrates an efficient approach to building an empathetic chatbot using LoRA fine-tuning on a lightweight model, suitable for resource-constrained environments and rapid prototyping.
