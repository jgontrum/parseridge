import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, method, hidden_size, device="cpu"):
        super(Attention, self).__init__()

        self.method = method
        self.hidden_size = hidden_size
        self.device = device

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def forward(self, hidden, encoder_outputs):
        hidden = hidden[0].transpose(0, 1)
        hidden = hidden[:, -1, :]

        this_batch_size = encoder_outputs.size(0)
        max_len = encoder_outputs.size(1)

        # Create variable to store attention energies
        attn_energies = torch.zeros(
            this_batch_size, max_len, requires_grad=True)
        attn_energies = attn_energies.to(self.device)

        # For each batch of encoder outputs
        for b in range(this_batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):
                current_hidden = hidden[b]
                current_outputs = encoder_outputs[b, i]
                attn_energies[b, i] = self.score(
                    current_hidden, current_outputs
                )

        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_output):

        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy

        elif self.method == 'concat':
            concatenated = torch.cat((hidden, encoder_output))
            energy = self.attn(concatenated)
            energy = self.v.dot(energy)
            return energy
