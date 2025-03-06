import torch
import torch.nn as nn
import math

# 1. 멀티헤드 어텐션 클래스
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0  # d_model이 num_heads로 나누어 떨어져야 함
        self.d_k = d_model // num_heads  # 각 헤드의 차원
        self.num_heads = num_heads
        self.d_model = d_model

        # 쿼리, 키, 값, 출력에 대한 선형 변환
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)  # 스케일링 factor

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # 선형 변환 후 헤드로 분할: [batch_size, seq_len, num_heads, d_k]
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 어텐션 스코어 계산
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [batch_size, num_heads, seq_len, seq_len]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # 마스크 적용
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # 값에 어텐션 가중치 적용
        context = torch.matmul(attn, V)  # [batch_size, num_heads, seq_len, d_k]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # 다시 결합
        output = self.W_o(context)
        return output, attn

# 2. 피드포워드 네트워크 클래스
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# 3. 인코더 레이어 클래스
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 멀티헤드 어텐션
        attn_output, _ = self.mha(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))  # 잔차 연결 + 정규화

        # 피드포워드
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))  # 잔차 연결 + 정규화
        return x

# 4. 디코더 레이어 클래스
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.enc_dec_mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # 셀프 어텐션
        self_attn_output, _ = self.self_mha(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))

        # 인코더-디코더 어텐션
        enc_dec_output, _ = self.enc_dec_mha(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(enc_dec_output))

        # 피드포워드
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))
        return x

# 5. 위치 인코딩 클래스
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

# 6. 트랜스포머 전체 모델 클래스
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048, dropout=0.1, max_len=5000):
        super(Transformer, self).__init__()
        self.d_model = d_model

        # 임베딩과 위치 인코딩
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # 인코더와 디코더 스택
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        # 출력 레이어
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, src_len]
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)  # [batch_size, 1, tgt_len, 1]
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask.to(tgt.device)
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        # 입력 임베딩과 위치 인코딩
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoding(src)
        src = self.dropout(src)

        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoding(tgt)
        tgt = self.dropout(tgt)

        # 인코더
        enc_output = src
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        # 디코더
        dec_output = tgt
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        # 출력
        output = self.fc_out(dec_output)
        return output