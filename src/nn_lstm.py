import torch
import torch.nn as nn


class LSTMNet(nn.Module):

    def __init__(self, n_hidden, device, num_lstm_stack_layers):
        super(LSTMNet, self).__init__()
        self.n_hidden = n_hidden
        self.device = device
        self.num_lstm_stack_layer = num_lstm_stack_layers

        # LSTM (input_features, hidden_features, stack_layers)
        self.lstm = nn.LSTM(4, self.n_hidden, self.num_lstm_stack_layer)
        self.linear = nn.Linear(self.n_hidden, 4)


    def forward(self, x, future=0):
        outputs = [None] * (future + 1)
        n_samples = x.size(0)

        h0 = torch.zeros(self.num_lstm_stack_layer, n_samples, self.n_hidden, dtype=torch.float32).to(self.device)
        c0 = torch.zeros(self.num_lstm_stack_layer, n_samples, self.n_hidden, dtype=torch.float32).to(self.device)

        input_tensor = x.permute(2, 0, 1)

        output, (hn, cn) = self.lstm(input_tensor, (h0, c0))
        output = self.linear(output)            # output.shape = (sequence_length, batch_size, features)
        # outputs.append(output)
        outputs[0] = output

        # Take the last pred in the sequence to make future predictions
        # we use "-2:-1" instead of "-1" because we need to keep 3d tensor:
        last_output_t = output[-2:-1, :, :]

        for i in range(future):
            last_output_t, (hn, cn) = self.lstm(last_output_t, (hn, cn))
            last_output_t = self.linear(last_output_t)
            outputs[i + 1] = last_output_t

        # outputs = torch.stack(outputs, dim=0)   # concats tensors with a new dimension
        outputs = torch.cat(outputs, dim=0)
        outputs = outputs.permute(1, 2, 0)  # the shape becomes :  (batch_size, features, seq_len)
        return outputs


    def count_params(self):
        """ Counts number of parameters in the network exposed to gradient optimization """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

