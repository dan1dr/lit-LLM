import tiktoken
import torch

def generate_text_simple(model, idx, max_new_tokens, context_size): # idx is a (batch, n_tokens) tensor
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:] # crops current context if it exceeds context size
        with torch.no_grad():
            logits = model(idx_cond) # model generates logits for the next token

        logits = logits[:, -1, :] # selects the last token in the sequence: (batch, n_tokens, vocab_size) 
                                    #-> (batch, vocab_size)
        probas = torch.softmax(logits, dim=-1) # converts logits to probabilities
        idx_next = torch.argmax(probas, dim=-1, keepdim=True) # selects the most likely token (batch, 1)
        idx = torch.cat([idx, idx_next], dim=-1) # appends the new token to the sequence, where idx has shape (batch, n_tokens+1)
    return idx

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special = {'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dim
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dim
    return tokenizer.decode(flat.tolist())