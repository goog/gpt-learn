import numpy as np



def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def layer_norm(x, g, b, eps: float = 1e-5):

    #print(f"x {x}  {x.shape}")
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(variance + eps)  # normalize x to have mean=0 and var=1 over last axis
    return g * x + b  # scale and offset with gamma/beta params

def linear(x, w, b):  # [m, in], [in, out], [out] -> [m, out]
    return x @ w + b

# process the output from one attention layer
# and fit the input for the next attention layer
def ffn(x, c_fc, c_proj):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # project up
    a = gelu(linear(x, **c_fc))  # [n_seq, n_embd] -> [n_seq, 4*n_embd]

    # project back down
    x = linear(a, **c_proj)  # [n_seq, 4*n_embd] -> [n_seq, n_embd]

    return x



def attention(q, k, v, mask):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v

def causal_self_attention(x, c_attn, c_proj): # [n_seq, n_embd] -> [n_seq, n_embd]
    # qkv projections
    print(x.shape)
    x = linear(x, **c_attn) # [n_seq, n_embd] -> [n_seq, 3*n_embd]
    print(f"after linear {x.shape}")

    # split into qkv
    q, k, v = np.split(x, 3, axis=-1) # [n_seq, 3*n_embd] -> 3 of [n_seq, n_embd]

    # causal mask to hide future inputs from being attended to
    causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10

    # perform causal self attention
    x = attention(q, k, v, causal_mask) # [n_seq, n_embd] -> [n_seq, n_embd]

    # out projection
    x = linear(x, **c_proj) # [n_seq, n_embd] @ [n_embd, n_embd] = [n_seq, n_embd]

    return x

def mha(x, c_attn, c_proj, n_head):
    x = linear(x, **c_attn)
    qkv = np.split(x, 3, axis=-1)
    #
    qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), qkv))
    causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10 

    out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]

    x = np.hstack(out_heads)  # [n_head, n_seq, n_embd/n_head] -> [n_seq, n_embd]

    # out projection
    x = linear(x, **c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x

def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)
    x = x + ffn(layer_norm(x, **ln_2), **mlp)

    return x


def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):
    print(f"wpe pos-encoding {wpe}, ")
    print(f"n_head {n_head}, ")
    print(f"inputs {len(inputs)} ,")
    print(f"wte[inputs] {wte[inputs]}")
    x = wte[inputs] +  wpe[range(len(inputs))]
    
    print("x shape:", x.shape)
    # forward pass through n_layer transformer blocks
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # projection to vocab
    x = layer_norm(x, **ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x @ wte.T


def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm
    
    for _ in tqdm(range(n_tokens_to_generate), "generating"):  # auto-regressive decode loop
        logits = gpt2(inputs, **params, n_head=n_head)  # model forward pass
        next_id = np.argmax(logits[-1])  # greedy sampling
        inputs.append(int(next_id))  # append prediction to input

    return inputs[len(inputs) - n_tokens_to_generate :] 
    
    
    
    
def main(prompt: str, n_tokens_to_generate: int = 40, model_size: str = "124M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params
    # load encoder, hparams, and params from the released open-ai gpt-2 files
    # hparams: config
    # params:  weights
    # encoder: bpe tokenizer
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)
    
    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)
    
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]
    
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)
    
    # decode tokens back into a string
    output_text = encoder.decode(output_ids)

    return output_text
    
    

if __name__ == "__main__":
    import fire

    fire.Fire(main)
