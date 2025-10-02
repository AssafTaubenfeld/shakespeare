def top_k_sample(logits, k=50):
    values, indices = torch.topk(logits, k)
    probs = torch.softmax(values, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return indices[next_token]

def pipeline(model, tokenizer, k=10):
    """    Create a text generation pipeline for the given model and tokenizer.
    Args:
        model: The trained model instance
        tokenizer: The tokenizer instance
        k: Number of top-k tokens to sample from at each step (default: 10)
    Returns:A function that takes a prompt and generates text"""
    # step-by-step explanation of the pipeline function:
    """
    1. Set the model to evaluation mode
    2. Encode the input prompt using the tokenizer
    3. Convert the encoded prompt to a tensor and move it to the model's device
    4. While the length of the generated sequence is less than max_len:
        a. Get the model's logits for the last token in the sequence
        b. Extract the logits for the last token and apply softmax to get probabilities
        v. Use top-k sampling to select the next token based on probabilities
        d. Append the selected token to the sequence
    5. Decode the generated sequence back to text using the tokenizer
    6. Return the generated text
    """
    def generate(prompt, max_len=1024):

        model.eval()
        device = next(model.parameters()).device 
        encoded_prompt = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)
        encoded_prompt = encoded_prompt.unsqueeze(0)
        
        max_new_tokens = min(max_len, model.config.block_size - encoded_prompt.size(1) - 1)

        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits, _ = model(encoded_prompt)
                next_token_logits = logits[0, -1, :]
                next_token = top_k_sample(next_token_logits, k=10)

                # Update the squence with the new token
                encoded_prompt = torch.cat([encoded_prompt, next_token.unsqueeze(0)], dim=1)
        
        return tokenizer.decode(encoded_prompt[0].tolist())
    
    return generate 


# exmple usage of the pipeline
"""
generate_text = pipeline(model, tokenizer)
prompt = "To be or not to be, that is the question: "
generated_text = generate_text(prompt, max_len=100)
"""
