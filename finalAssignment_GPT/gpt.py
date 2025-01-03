import torch
import torch.nn as nn
from torch.nn import functional as F
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 384 # what is the maximum context length for predictions?
max_iters = 10
eval_interval = 200
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 5
n_embd = 120
n_head = 6
n_layer = 2
dropout = 0.2
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('inputMelodies.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
print(f"Chars: {chars}")
vocab_size = len(chars)
print(f'Vocab size: {vocab_size}')
print(f'Text size (chars): {len(text)}')
print(f'Text size (words): {len(text.split())}')
print(f'Unique words: {len(set(text.split()))}')

with open('inputMelodies.txt', 'r', encoding='utf-8') as f:
    child_speech_test_text = f.read()

with open('inputMelodiesAugmented.txt', 'r', encoding='utf-8') as f:
    shakespeare_test_text = f.read()

all_text = text + child_speech_test_text + shakespeare_test_text
chars = sorted(list(set(all_text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)

# Recreate the encode and decode functions
encode = lambda s: [stoi.get(c, stoi.get('<unk>')) for c in s]  # Map missing characters to <unk>
decode = lambda l: ''.join([itos[i] for i in l])

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]
print(f"Train data shape: {train_data.shape}, Val data shape: {val_data.shape}")

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            print(f"Evaluating loss for {split}, iter {k}")  # Debug print
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
        print(f"Average {split} loss: {out[split]:.4f}")  # Debug print
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=True)
        self.query = nn.Linear(n_embd, head_size, bias=True)
        self.value = nn.Linear(n_embd, head_size, bias=True)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    print(f"Starting iteration {iter}")
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))



child_speech_test_data = torch.tensor(encode(child_speech_test_text), dtype=torch.long)
shakespeare_test_data = torch.tensor(encode(shakespeare_test_text), dtype=torch.long)

print(f"Child Speech Test Data: {child_speech_test_data.shape}")
print(f"Shakespeare Test Data: {shakespeare_test_data.shape}")


@torch.no_grad()
def calculate_test_loss_and_perplexity(test_data):
    model.eval()
    total_loss = 0
    batch_count = 0

    for i in range(0, len(test_data) - block_size, block_size):
        x = test_data[i:i + block_size].unsqueeze(0).to(device)
        y = test_data[i + 1:i + block_size + 1].unsqueeze(0).to(device)

        _, loss = model(x, y)
        total_loss += loss.item()
        batch_count += 1

    average_loss = total_loss / batch_count
    perplexity  = torch.exp(loss)
    model.train()
    return average_loss, perplexity

def calculate_bleu(reference_text, generated_text):
    reference_tokens = [reference_text[:500].split()]
    generated_tokens = generated_text.split()
    smoothie = SmoothingFunction().method1
    return sentence_bleu(reference_tokens, generated_tokens, smoothing_function=smoothie)



#def dummy_model_loss(test_data):
#    vocab_size = len(chars)
#    dummy_loss = -torch.log(torch.tensor(1 / vocab_size)).item()  # Uniform probabilities
#    perplexity = torch.exp(dummy_loss)
#    return dummy_loss, perplexity

print("Evaluating on inputMelodies.txt")
InputMelody_test_loss, InputMelody_test_perplexity = calculate_test_loss_and_perplexity(child_speech_test_data)
print(f"Input Melodies Test Loss and perplexity -\nLoss: {InputMelody_test_loss:.4f}\nPerplexity: {InputMelody_test_perplexity:.4f}")

#print("Evaluating on inputMelodiesAugmented.txt")
#InputMelody_test_loss, InputMelody_test_perplexity = calculate_test_loss_and_perplexity(shakespeare_test_data)
#print(f"Augmented Melodies Test Loss and perplexity -\nLoss: {InputMelody_test_loss:.4f}\nPerplexity: {InputMelody_test_perplexity:.4f}")

# Comparison with dummy model
#print("\nComparing with a dummy model...")

context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_sequence = decode(model.generate(context, max_new_tokens=500)[0].tolist())

print(generated_sequence)
print(shakespeare_test_text)
bleu_score = calculate_bleu(shakespeare_test_text, generated_sequence)
print(f"Generated Text: {generated_sequence}")
print(f"BLEU Score: {bleu_score:.4f}")

#dummy_loss = dummy_model_loss(child_speech_test_data)
#print(f"Dummy Model child speech Loss: {dummy_loss:.4f}")
#dummy_loss = dummy_model_loss(shakespeare_test_data)
#print(f"Dummy Model shakespeare Loss: {dummy_loss:.4f}")

def test_bleu_function():
    reference_text = "R R R R R R f g f F d c g a g f a a a g f g f d d"
    generated_text = "R R R R f g f F d c g a g f a g f g f d d"

    bleu_score = calculate_bleu(reference_text, generated_text)
    print(f"Reference: {reference_text}")
    print(f"Generated: {generated_text}")
    print(f"BLEU Score: {bleu_score:.4f}")

# Run the test
test_bleu_function()

def debug_bleu(reference_text, generated_text):
    reference_tokens = [reference_text.split()]
    generated_tokens = generated_text.split()
    print("Reference Tokens:", reference_tokens)
    print("Generated Tokens:", generated_tokens)

debug_bleu(shakespeare_test_text, generated_sequence)
