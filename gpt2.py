import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
batch_size = 64 # num of independent sequqnces that gets processed
block_size = 256  #aka context length/each sequence length
max_iters = 5000
eval_iters = 200
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embed_dim = 384
num_heads = 6
num_layers = 6
dropout = 0.2

torch.manual_seed(42)

with open('input.txt', 'r', encoding='utf8') as f:
    text = f.read()

# get all the unique characters that occur in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# tokenizer stuff
stoi = { ch: i for i, ch in enumerate(chars) }
itos = { i: ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[char] for char in s] # encode a string to list of integers
decode = lambda l: ''.join(itos[i] for i in l) # inverse of encode function

# train and test splits

# encoding entire text
data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data) * .9)
train_data = data[:n]
valid_data = data[n:]

# data loader
def get_batch(split):
    data = train_data if split == "train" else valid_data
    random_indices = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[id: id+block_size] for id in random_indices], dim=0)
    y = torch.stack([data[id+1: id+block_size+1] for id in random_indices], dim=0)
    # moving x and y to 'device' hyperparam
    x, y = x.to(device=device), y.to(device=device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    # eval mode
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    # putting model back to train mode
    model.train()
    return out


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        head_size = embed_dim // num_heads
        self.multihead_attn = MultiHeadAttention(num_heads, head_size)
        self.ffn = FeedForwardNetwork(embed_dim)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # residual/skip connections employed and layernorm before passing ot submodules
        x = x + self.multihead_attn(self.layer_norm1(x))
        x = x + self.ffn(self.layer_norm2(x))
        return x


class FeedForwardNetwork(nn.Module):
    """This layer is usuaully considered as computional phase of the tokens"""
    def __init__(self, embed_dim):
        super(FeedForwardNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # [B, T, C]
        return self.network(x)


class MultiHeadAttention(nn.Module):
    """mulitple heads of self-attention in parallel"""
    def __init__(self, num_heads, head_size):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # concatenate all the heads
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class Head(nn.Module):
    """a single head of self-attention"""
    def __init__(self, head_size):
        super(Head, self).__init__()
        self.key = nn.Linear(embed_dim, head_size)
        self.query = nn.Linear(embed_dim, head_size)
        self.value = nn.Linear(embed_dim, head_size)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape

        K = self.key(x)
        Q = self.query(x)
        # compute attentions scores
        attn_scores = K @ Q.transpose(-2, -1) * C**-0.5 # [B,T,C] @ [B,C,T] => [B,T,T]
        # mask the values for further positions
        mask = self.tril[:T,:T] == 0
        attn_scores = attn_scores.masked_fill(mask, float('-inf')) # [B,T,T]
        # softmax
        attn_scores = F.softmax(attn_scores,dim=-1) # [B,T,T]
        attn_scores = self.dropout(attn_scores)
        V = self.value(x)
        out = attn_scores @ V # [B, T, T] @ [B, T, head_size] => [B, T, head_size]
        return out

# model 
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super(BigramLanguageModel, self).__init__()
        # embeddings table for lookup
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding_table = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.Sequential(*[Block(embed_dim, num_heads=num_heads) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx, targets=None): # targets is optional here
        # idx(input) and targets are both tensors of shape [B, T]
        # turn each token into a vector
        B, T = idx.shape

        token_embed = self.token_embedding_table(idx) # after lookup the token_embed shape would be [B, T, C] => [batch_size, block_size, dim_size]
        pos_embed = self.position_embedding_table(torch.arange(T, device=device))
        x = token_embed + pos_embed # [B,T,C]
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.lm_head(x) # [B, T, vocab_size]

        if targets is None:
            loss = None
        else:
            # reshape idx and targets to match cross_entropy's requirements
            B, T, C = logits.shape

            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # calculate loss bw preds and actual target values
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # generate adds new token to dim=1(Time dimension) upto max_tokens value.
        # idx shape would be [B, T] and it's converted to => [B, T+1] => [B, T+2].. until max_tokens

        for i in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # call forward to get the loss and logits
            logits, loss = self(idx_cond)
            # get last time dimension value which is the prediction?
            logits = logits[:, -1, :] # [B, C]

            # get prob distribution of this
            probs = F.softmax(logits, dim=-1) # [B, C]

            idx_next = torch.multinomial(probs, num_samples=1) # [B, 1] => a column vector
            # concatenate to idx
            idx = torch.cat((idx, idx_next), dim=1) # [B, T+1]
        return idx


model = BigramLanguageModel()
# moving model to 'deivce'
model.to(device=device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# training loop
for iter in range(max_iters):
    # evalulate loss on train and val sets every once in a while
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # sample batch of data
    xb, yb = get_batch('train')

    # model predictions
    logits, loss = model(xb, yb)
    # flush out previous graidents
    optimizer.zero_grad()
    # calculate gradients for this iteration
    loss.backward()
    # update weights using calculated grads
    optimizer.step()


# generate from model
# context1 = torch.tensor([[1]], dtype=torch.long, device=device)
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
