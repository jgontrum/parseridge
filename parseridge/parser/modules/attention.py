import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, method, hidden_size, device="cpu"):
        super(Attention, self).__init__()
        self.device = device

        self.method = method
        self.hidden_size = hidden_size

        self.attn = nn.Linear(self.hidden_size + 64, self.hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))


    def forward(self, hidden, encoder_outputs, src_len=None):
        """
        :param hidden:
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :param src_len:
            used for masking. NoneType or tensor in shape (B) indicating sequence length
        :return
            attention energies in shape (B,T)
        """
        max_len = encoder_outputs.size(1) # CHECK dims
        this_batch_size = encoder_outputs.size(0) # CHECK dims
        hidden = hidden.transpose(0, 1)
        H = hidden.repeat(max_len, 1, 1).transpose(0, 1) # CHECK dims
        attn_energies = self.score(H, encoder_outputs)  # compute attention score

        if src_len is not None:
            mask = []
            for b in range(len(src_len)):
                mask.append(
                    [0] * src_len[b] + [1] * (encoder_outputs.size(1) - src_len[b])
                )

            mask = torch.ByteTensor(mask, device=self.device)
            attn_energies = attn_energies.masked_fill(mask, -1e18)

        return F.softmax(attn_energies, dim=0).unsqueeze(1)  # normalize with softmax


    def score(self, hidden, encoder_outputs):
        merged = torch.cat((hidden, encoder_outputs), 2)
        energy = torch.tanh(
            self.attn(merged)
        )  # [B*T*2H]->[B*T*H]

        energy = energy.transpose(2, 1)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]
