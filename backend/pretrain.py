import math
import re
from random import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import spacy


nlp = spacy.load("en_core_web_sm")
n_layers = 6    # number of Encoder of Encoder Layer
n_heads  = 8    # number of heads in Multi-Head Attention
d_model  = 768  # Embedding Size
d_ff = 768 * 4  # 4*d_model, FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_segments = 2
vocab_size = 86
batch_size = 6
max_mask   = 5  
max_len    = 300

class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
        self.pos_embed = nn.Embedding(max_len, d_model)      # position embedding
        self.seg_embed = nn.Embedding(n_segments, d_model)  # segment(token type) embedding
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        # print("Input Shape - x:", x.shape, "seg:", seg.shape)
        # print("Input Indices Range:")
        # print("Min Input ID:", torch.min(x))
        # print("Max Input ID:", torch.max(x))
        # print("Min Segment ID:", torch.min(seg))
        # print("Max Segment ID:", torch.max(seg))

        # print("\nVocabulary Sizes:")
        # print("Token Embedding Vocabulary Size:", self.tok_embed.num_embeddings)
        # print("Segment Embedding Vocabulary Size:", self.seg_embed.num_embeddings)

        #x, seg: (bs, len)
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)  # (len,) -> (bs, len)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        # print("Shape of Token Embedding:", self.tok_embed(x).shape)
        # print("Shape of Position Embedding:", self.pos_embed(pos).shape)
        # print("Shape of Segment Embedding:", self.seg_embed(seg).shape)
        # print("Shape of Combined Embedding:", embedding.shape)
        return self.norm(embedding)
    



def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn       = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn
    


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn
    


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = nn.Linear(n_heads * d_v, d_model)(context)
        return nn.LayerNorm(d_model)(output + residual), attn # output: [batch_size x len_q x d_model]


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, len_seq, d_model) -> (batch_size, len_seq, d_ff) -> (batch_size, len_seq, d_model)
        return self.fc2(F.gelu(self.fc1(x)))
    


class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, d_model)
        self.activ = nn.Tanh()
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 2)


        # decoder is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, input_ids, segment_ids, masked_pos):
        output = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
        for layer in self.layers:
            output, enc_self_attn = layer(output, enc_self_attn_mask)
        # output : [batch_size, len, d_model], attn : [batch_size, n_heads, d_mode, d_model]

        # 1. predict next sentence
        # it will be decided by first token(CLS)
        h_pooled   = self.activ(self.fc(output[:, 0])) # [batch_size, d_model]
        logits_nsp = self.classifier(h_pooled) # [batch_size, 2]

        # 2. predict the masked token
        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1)) # [batch_size, max_pred, d_model]
        h_masked = torch.gather(output, 1, masked_pos) # masking position [batch_size, max_pred, d_model]
        h_masked  = self.norm(F.gelu(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias # [batch_size, max_pred, n_vocab]

        return logits_lm, logits_nsp, h_pooled





def preprocess_text(raw_text):
    #nlp = spacy.load("en_core_web_sm")
    doc_new = nlp(raw_text)
    sentences_new = list(doc_new.sents)
    #print(f'sentences_new:{len(sentences_new)}')

    # Lower case and clean all the symbols
    text_new = [x.text.lower() for x in sentences_new]
    text_new = [re.sub("[.,!?\\-]", '', x) for x in text_new]

    # Making vocabs - numericalization
    word_list_new = list(set(" ".join(text_new).split()))
    word2id_new = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}

    for i, w in enumerate(word_list_new):
        word2id_new[w] = i + 4  # reserve the first 0-3 for CLS, PAD
    id2word_new = {i: w for i, w in enumerate(word2id_new)}
    vocab_size_new = len(word2id_new)

    # print(f"id2word_new:{id2word_new}")
    token_list_new = list()
    for sentence in sentences_new:
        arr = [word2id_new[word] for sentence in text_new for word in sentence.split()]
        token_list_new.append(arr)
    # print(f'token_list_new length:{len(token_list_new)}')
    # print(f'token_list_new:{token_list_new}')

    return token_list_new, word2id_new, id2word_new, vocab_size_new





def make_batch(sent, batch_size, max_mask, max_len):
    token_list, word2id, id2word, vocab_size = preprocess_text(" ".join(sent))

    #print(f'batch size:{batch_size}')

    batch = []
    positive = negative = 0

    while positive != batch_size // 2 or negative != batch_size // 2:
        tokens_a_index, tokens_b_index = randrange(len(token_list)), randrange(len(token_list))
        # print(f'tokens_a_index:{tokens_a_index}')
        # print(f'tokens_b_index:{tokens_b_index}')
        tokens_a, tokens_b = token_list[tokens_a_index], token_list[tokens_b_index]

        input_ids = [word2id['[CLS]']] + tokens_a + [word2id['[SEP]']] + tokens_b + [word2id['[SEP]']]
        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)

        n_pred = min(max_mask, max(1, int(round(len(input_ids) * 0.15))))
        candidates_masked_pos = [i for i, token in enumerate(input_ids) if token != word2id['[CLS]'] and token != word2id['[SEP]']]
        shuffle(candidates_masked_pos)
        masked_tokens, masked_pos = [], []

        # print('here')

        for pos in candidates_masked_pos[:n_pred]:
            # print('for')
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            if random() < 0.1:
                index = randint(0, vocab_size - 1)
                input_ids[pos] = word2id[id2word[index]]
            elif random() < 0.8:
                input_ids[pos] = word2id['[MASK]']
            else:
                pass

        n_pad = max_len - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)

        if max_mask > n_pred:
            n_pad = max_mask - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)

        if tokens_a_index + 1 == tokens_b_index and positive < batch_size / 2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True])
            positive += 1
        elif tokens_a_index + 1 != tokens_b_index and negative < batch_size / 2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False])
            negative += 1

    return batch




def preprocess_function(examples):
    max_mask = 5
    max_len = 300
    # Convert examples to list of sentences
    sents = [f"{premise} {hypothesis}" for premise, hypothesis in zip(examples["premise"], examples["hypothesis"])]
    # Generate batch
    batch = make_batch(sents, len(sents), max_mask, max_len)
    return {
        "input_ids": torch.tensor([example[0] for example in batch]),
        "segment_ids": torch.tensor([example[1] for example in batch]),
        "masked_tokens": torch.tensor([example[2] for example in batch]),
        "masked_pos": torch.tensor([example[3] for example in batch]),
        "labels": torch.tensor([example[4] for example in batch], dtype=torch.long)
    }






# Load the pickled models
try:
    with open('./files/id2word.pickle', 'rb') as f:
        id2word = pickle.load(f)
        # print('id2word successfully loaded')

    with open('./files/word2id.pickle', 'rb') as f:
        word2id = pickle.load(f)
        # print('word2id successfully loaded')
    
    with open('./files/tokenList.pickle', 'rb') as f:
        token_list = pickle.load(f)
        # print('token_list successfully loaded')

except FileNotFoundError as e:
    print(f'Error loading pickled models: {e}')




# Load pytorch models
try:
    path_pytorch = './files/'
    # load pretrained BERT
    loaded_model = BERT()  # Assuming BERT is defined somewhere in your code
    loaded_model.load_state_dict(torch.load(path_pytorch + 'bert_model.pth'))
    loaded_model.eval()  #


    # load classifier
    finetuned_model = BERT()  # Assuming BERT is defined somewhere in your code
    finetuned_model.load_state_dict(torch.load(path_pytorch + 'fine_tuned_model.pth'))
    finetuned_model.eval()  #

except FileNotFoundError as e:
    print(f'Error loading pytorch models: {e}')




# print(f'loaded_model:{loaded_model}')
# print(f'finetuned_model:{finetuned_model}')





class Model:
    def __init__(self, sentence_a, sentence_b):
        self.word2id = word2id
        self.id2word = id2word
        self.loaded_model = loaded_model
        self.finetuned_model = finetuned_model
        self.sentence_a = sentence_a
        self.sentence_b = sentence_b

        print('before')
        batch_a = make_batch(sentence_a, len(sentence_a), max_mask, max_len)
        batch_b = make_batch(sentence_b, len(sentence_b), max_mask, max_len)

        input_ids_a, segment_ids_a, masked_tokens_a, masked_pos_a, isNext_a = map(torch.LongTensor, zip(*batch_a))
        input_ids_b, segment_ids_b, masked_tokens_b, masked_pos_b, isNext_b = map(torch.LongTensor, zip(*batch_b))

        _, _, self.u = self.loaded_model(input_ids_a, segment_ids_a, masked_pos_a)
        _, _, self.v = self.loaded_model(input_ids_b, segment_ids_b, masked_pos_b)

        print('after')


    def cosine_similarity_scratch(self):
        print('yes here')
        dot_product = torch.dot(self.u.flatten(), self.v.flatten()).item()  # Convert to scalar
        norm_u = torch.norm(self.u)
        norm_v = torch.norm(self.v)
        return dot_product / (norm_u * norm_v)

