import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerModel(nn.Module):
    def __init__(
            self,
            vocab_size,
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1
    ):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )

        # Output layer
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Model hyperparameters
        self.d_model = d_model

    def generate_square_subsequent_mask(self, sz):
        """
        Generate a mask to prevent looking at future tokens during training
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, src, tgt):
        # Embed and add positional encoding
        src = self.embedding(src) * math.sqrt(self.d_model)
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)

        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        # Create attention masks
        src_mask = None
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        # Encode source sequence
        memory = self.transformer_encoder(src.transpose(0, 1))

        # Decode target sequence
        output = self.transformer_decoder(
            tgt.transpose(0, 1),
            memory,
            tgt_mask=tgt_mask
        )

        # Generate output
        output = self.fc_out(output.transpose(0, 1))

        return output

    def inference(self, src, max_len, start_token, end_token):
        """
        Inference method for generating sequences
        """
        self.eval()  # Set to evaluation mode

        # Encode source sequence
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.positional_encoding(src)
        memory = self.transformer_encoder(src.transpose(0, 1))

        # Initialize generation
        generated = [start_token]

        with torch.no_grad():
            for _ in range(max_len):
                # Convert generated sequence to tensor
                tgt = torch.tensor(generated).unsqueeze(0).to(src.device)

                # Embed and prepare target
                tgt = self.embedding(tgt) * math.sqrt(self.d_model)
                tgt = self.positional_encoding(tgt)

                # Generate mask
                tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

                # Decode
                output = self.transformer_decoder(
                    tgt.transpose(0, 1),
                    memory,
                    tgt_mask=tgt_mask
                )

                # Get last token prediction
                pred = self.fc_out(output.transpose(0, 1)[:, -1, :])
                next_token = pred.argmax(dim=-1).item()

                # Append and check for end
                generated.append(next_token)
                if next_token == end_token:
                    break

        return generated

