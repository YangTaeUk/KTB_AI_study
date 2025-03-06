import torch
import torch.nn as nn
import math

class PositinalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len = 5000):
        """
        d_model: 임베딩 차원
        dropout: dropout 비율
        max_len: 최대 시퀀스 길이
        """
        super(PositinalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 고정된 포지셔널 인코딩 행렬 생성, 위치벡터 생성
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)


    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1):
        """
        vocab_size: 어휘 사전의 크기
        d_model: 임베딩 차원
        nhead: 멀티헤드 어텐션 헤드 수
        num_encoder_layers: 인코더 레이어 수
        num_decoder_layers: 디코더 레이어 수
        dim_feedforward: 피드포워드 네트워크 모델 차원
        dropout: 드롭아웃 비율
        """
        super(TransformerModel, self).__init__()
        self.d_model = d_model

        # 입력 임베딩 및 포지셔널 인코딩
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositinalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(d_model, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward, dropout=dropout)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        """
        src: (seq_len_src, batch_size)
        tgt: (seq_len_tgt, batch_size)
        주의: nn.Transformer는 기본적으로 (seq_len, batch_size, d_model) 형태를 사용합니다.
        """
        src_emb = self.embedding(src) * math.sqrt(self.d_model)  # (seq_len_src, batch_size, d_model)
        src_emb = self.pos_encoder(src_emb.transpose(0, 1)).transpose(0, 1)  # 유지: (seq_len_src, batch_size, d_model)
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)  # (seq_len_tgt, batch_size, d_model)
        tgt_emb = self.pos_encoder(tgt_emb.transpose(0, 1)).transpose(0, 1)  # (seq_len_tgt, batch_size, d_model)

        output = self.transformer(src_emb, tgt_emb, src_mask=src_mask,
                                  tgt_mask=tgt_mask, memory_mask=memory_mask)
        output = self.fc_out(output)  # (seq_len_tgt, batch_size, vocab_size)
        return output
