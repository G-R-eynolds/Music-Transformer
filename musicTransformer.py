import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from Transformer import InputEmbeddings, LayerNormalization, FeedForwardBlock, MultiHeadAttention, ResidualConnection, LastLinear, DecoderBlock, Encoder, Decoder, Transformer, PositionalEmbeddings

class RelativeMultiHeadAttention(MultiHeadAttention):
    def __init__(self, n_heads, d_model, max_dist, dropout):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads
        self.dropout = nn.Dropout(dropout)

        #initialise weight matrices
        self.w_q = nn.Linear(d_model, d_model, bias = False)  #by using a Linear layer we can speed up the calculations via PyTorch, and setting bais to False makes this just a weights matrix as we want
        self.w_k = nn.Linear(d_model, d_model, bias = False)
        self.w_v = nn.Linear(d_model, d_model, bias = False)

        #initialise W_0:
        self.w_0 = nn.Linear(d_model, d_model)

        #this is the new bit: we need to initialise the relative positional encodings
        self.max_dist = max_dist
        self.rel_embeddings = nn.Parameter(
            torch.randn(n_heads, max_dist + 1, self.d_k)  #we add 1 to max_dist as we need to include the 0th position
        )

    def rel_attention(self, q):
        #plan to implement the "skewing" algorithm as described in the paper
        batch_size, n_heads, seq_len, d_k = q.size()
        effective_dist = min(self.max_dist, seq_len - 1) #no need to consider distances longer than the sequence

        q_er = torch.einsum('bhid,hrd->bhir', q, self.rel_embeddings[:, :effective_dist+1]) #einstein summation of matrix multiplication, this is essentially shorthand for:
        #(batch_size, n_heads, seq_len, d_k) x (n_heads, effective_dist+1, d_k) -> (batch_size, n_heads, seq_len, effective_dist+1)

        padded = F.pad(q_er, (1,0))  #pad the tensor with 0s on the left side

        batch_size, n_heads, seq_len, rel_dist_plus_1 = padded.size()
        padded = padded.view(batch_size, n_heads, rel_dist_plus_1, seq_len) #reshape to fit

        rel_scores = padded[:, :, 1:, :].view(batch_size, n_heads, seq_len, seq_len) #slice and reshape

        return rel_scores
    
    def attention(self, q, k ,mask, dropout: nn.Dropout):
        d_k = q.shape[3]
        content_scores = torch.matmul(q, k.transpose(-2,-1))/math.sqrt(d_k)
        rel_scores = self.rel_attention(q)
        scores = content_scores + rel_scores
        if mask is not None:
            scores.masked_fill(mask == 0, -1e9)
        if dropout is not None:
            scores = dropout(scores)
        return scores
    
    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        values = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.n_heads, self.d_k).transpose(1, 2)  
        key = key.view(key.shape[0], key.shape[1], self.n_heads, self.d_k).transpose(1, 2)
        values = values.view(values.shape[0], values.shape[1], self.n_heads, self.d_k).transpose(1, 2)
        #this is all the same as normal

        scores = self.attention(q, k, mask)

        probs = F.softmax(scores, dim = -1)
        if self.dropout is not None:
            probs = self.dropout(probs)
        x = torch.matmul(probs, values)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.n_heads * self.d_k)

        return self.w_0(x)
    
class DecoderOnlyBlock(nn.Module):
    #as generative transformers only need the decoder, this block is a simplified version of the Transformer block without cross attention
    def __init__(self, self_attention: RelativeMultiHeadAttention, feed_forward: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])  #creates a list with 3 residual connection modules

    def forward(self, x, mask1):
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, mask1))
        x = self.residual_connections[1](x, lambda x: self.feed_forward(x))
        return x
    
class DecoderOnly(nn.Module):
    #stack multiple DecoderOnlyBlocks as in the standard Transformer implementation
    def __init__(self, layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
#since we are only using the decoder, we need to create a new MusicTransformer class to only use the decoder
class MusicTransformer(nn.Module):
    def __init__(self, decoder: DecoderOnly, embed: InputEmbeddings, pos: PositionalEmbeddings, output: LastLinear):
        super().__init__()
        self.decoder = decoder
        self.embed = embed
        self.pos = pos
        self.output = output

    def forward(self, x, mask):
        x = self.embed(x)
        x = self.pos(x)
        x = self.decoder(x, mask)
        x = self.output(x)
        return x
    
    def generate(self, start_tokens, max_length=None, temperature=1.0):
        #start_tokens is a tensor of shape [batch_size, seq_len] containing an initial motif
        if max_length is None:
            max_length = start_tokens.size(1) * 2
        
        batch_size = start_tokens.size(0)
        device = start_tokens.device
        generated = start_tokens.clone() #starts building from a copy of the start tokens (i.e continuing the piece)
        
        for i in range(start_tokens.size(1), max_length):
            mask = torch.triu(torch.ones(i, i, device=device), diagonal=1).bool().unsqueeze(0).unsqueeze(1)  #creates casual mask
            
            with torch.no_grad():
                logits = self.forward(generated[:, :i], mask) #runs the forward method on all tokens generated so far
                
                next_token_logits = logits[:, -1, :] / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                
                next_token = torch.multinomial(probs, 1) #uses multinomial sampling to select the next token
                
                generated = torch.cat([generated, next_token], dim=1) #appends the new token to the generated sequence
        
        return generated
        
def build_music_transformer(vocab_size: int, seq_len: int, d_model: int = 512, n_layers: int = 6,n_heads: int = 8, dropout: float = 0.1, d_ff: int = 2048, max_rel_dist: int = 512):
    #function that mirrors the build_transformer function, except is simply as it builds a decoder-only transformer
    embed = InputEmbeddings(d_model, vocab_size)  
    pos = PositionalEmbeddings(d_model, seq_len, dropout)
    
    decoder_blocks = []
    for _ in range(n_layers):
        self_attention = RelativeMultiHeadAttention(d_model, n_heads, max_rel_dist, dropout)      
        feed_forward = FeedForwardBlock(d_model, d_ff, dropout)    
        decoder_block = DecoderOnlyBlock(self_attention, feed_forward, dropout)
        decoder_blocks.append(decoder_block)
    
    decoder = DecoderOnly(nn.ModuleList(decoder_blocks))
    
    output = LastLinear(d_model, vocab_size)
    
    model = MusicTransformer(decoder, embed, pos, output)
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return model