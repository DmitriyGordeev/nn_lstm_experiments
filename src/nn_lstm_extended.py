import torch
import torch.nn as nn

class LSTMExtended(nn.Module):
    def __init__(self, n_hidden, device, num_lstm_stack_layers, tgt_future_len):
        super(LSTMExtended, self).__init__()
        self.device = device

        self.lstm_features = 32      # Conv out features
        self.n_hidden = n_hidden
        self.num_lstm_stack_layer = num_lstm_stack_layers

        self.tgt_future_len = tgt_future_len
        self.lstm_future_out = 10


        # # 1. Convolution along time dimension
        self.time_conv = nn.Conv1d(in_channels=4, out_channels=self.lstm_features, kernel_size=30, stride=6)
        # self.linear_time_collpase = nn.Linear(150, 30)

        self.linear_src = nn.Linear(150, 49)        # to sum up with final output

        # 2. LSTM (input_features, hidden_features, stack_layers)
        self.lstm = nn.LSTM(self.lstm_features, self.n_hidden, self.num_lstm_stack_layer)
        self.lstm_decoder = nn.Linear(self.n_hidden, self.lstm_features)

        # self.linear_time_expand = nn.Linear(self.lstm_future_out, self.tgt_future_len)
        # self.linear_feature_collapse = nn.Linear(self.lstm_features, 1)

        self.deconv = nn.ConvTranspose1d(in_channels=self.lstm_features, out_channels=1, kernel_size=40, stride=1, padding=0, bias=False)


    def forward(self, x):
        x = self.time_conv(x)   # (batch_size, features, seq)
        x = torch.relu(x)       # (batch_size, features, seq)

        n_samples = x.size(0)

        h0 = torch.zeros(self.num_lstm_stack_layer, n_samples, self.n_hidden, dtype=torch.float32).to(self.device)
        c0 = torch.zeros(self.num_lstm_stack_layer, n_samples, self.n_hidden, dtype=torch.float32).to(self.device)

        x = x.permute(2, 0, 1)

        output, (hn, cn) = self.lstm(x, (h0, c0))
        output = self.lstm_decoder(output)            # output.shape = (sequence_length, batch_size, features)

        future_outputs = [None] * self.lstm_future_out

        # Take the last pred in the sequence to make future predictions
        # we use "-2:-1" instead of "-1" because we need to keep 3d tensor:
        last_output_t = output[-2:-1, :, :]

        for i in range(self.lstm_future_out):
            last_output_t, (hn, cn) = self.lstm(last_output_t, (hn, cn))
            last_output_t = self.lstm_decoder(last_output_t)
            future_outputs[i] = last_output_t
        future_outputs = torch.cat(future_outputs, dim=0)
        future_outputs = future_outputs.permute(1, 2, 0)  # the shape becomes :  (batch_size, features, fut_seq_len)

        # TODO: relu after lstm_decoder ?

        future_outputs = self.deconv(future_outputs)
        return future_outputs


    def count_params(self):
        """ Counts number of parameters in the network exposed to gradient optimization """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
