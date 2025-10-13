import sys
import os

from dgl.nn.pytorch.conv import GraphConv, GATConv
from torch_geometric.nn import GraphNorm, global_mean_pool, global_max_pool
from torch.nn import ReLU, GELU
import math

class GRU(torch.nn.Module):
    def __init__(self, in_channels, gru_size, hidden_channels, end_channels, n_sequences, device,
                 act_func='ReLU', task_type='regression', dropout=0.0, num_layers=1,
                 return_hidden=False, out_channels=None, use_layernorm=False, horizon=0):
        super(GRU, self).__init__()

        self.device = device
        self.return_hidden = return_hidden
        self.num_layers = num_layers
        self.hidden_size = hidden_channels
        self.task_type = task_type
        self.is_graph_or_node = False
        self.gru_size = gru_size
        self.end_channels = end_channels
        self.n_sequences = n_sequences
        self.decoder = None
        self._decoder_input = None
        self.horizon = horizon
        self.out_channels = out_channels

        # GRU layer
        self.gru = torch.nn.GRU(
            input_size=in_channels + self.end_channels if horizon > 0 else in_channels,
            hidden_size=gru_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        ).to(device)

        # Optional normalization layer
        if use_layernorm:
            self.norm = torch.nn.LayerNorm(gru_size).to(device)
        else:
            self.norm = torch.nn.BatchNorm1d(gru_size).to(device)

        # Dropout after GRU
        self.dropout = torch.nn.Dropout(p=dropout).to(device)

        # Output linear layer
        self.linear1 = torch.nn.Linear(gru_size, hidden_channels).to(device)
        self.linear2 = torch.nn.Linear(hidden_channels, end_channels).to(device)
        self.output_layer = torch.nn.Linear(end_channels, out_channels).to(device)

        # Activation function
        self.act_func = getattr(torch.nn, act_func)()

        # Output activation depending on task
        if task_type == 'classification':
            self.output_activation = torch.nn.Softmax(dim=-1).to(device)
        elif task_type == 'binary':
            self.output_activation = torch.nn.Sigmoid().to(device)
        else:
            self.output_activation = torch.nn.Identity().to(device)  # For regression or custom handling

        if self.horizon > 0:
            self.define_horizon_decodeur()

    def forward(self, X, edge_index=None, graphs=None, z_prev=None):
        """
        Parameters:
            X: Tensor of shape (batch_size, features, sequence_length)

        Returns:
            output: Final prediction tensor
            (optionally) hidden_repr: The hidden state before final layer
        """
        batch_size = X.size(0)

        if z_prev is None:
            z_prev = torch.zeros((X.shape[0], self.end_channels, self.n_sequences), device=X.device, dtype=X.dtype)
        else:
            z_prev = z_prev.view(X.shape[0], self.end_channels, self.n_sequences)
        
        if self.horizon > 0:
            x = torch.cat((X, z_prev), dim=1)

        # Reshape to (batch, seq_len, features)
        x = X.permute(0, 2, 1)

        # Initial hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.gru_size).to(self.device)

        # GRU forward
        x, _ = self.gru(x, h0)

        # Last time step output
        x = x[:, -1, :]  # shape: (batch_size, hidden_size)

        # Normalization and dropout
        x = self.norm(x)
        x = self.dropout(x)

        # Activation and output
        x = self.act_func(self.linear1(x))
        hidden = self.act_func(self.linear2(x))
        logits = self.output_layer(hidden)
        output = self.output_activation(logits)
        self._decoder_input = hidden
        return output, logits, hidden

    def define_horizon_decodeur(self, decodeur_params=None):
        if decodeur_params is None:
            if self.out_channels is None:
                raise ValueError("out_channels must be specified to automatically define the decoder.")
            if self.horizon <= 0:
                raise ValueError("horizon must be greater than zero to automatically define the decoder.")
            decodeur_params = {
                'device': self.device,
                'hidden_dim': self.end_channels,
                'output_dim': self.out_channels * self.horizon,
            }

        device = decodeur_params.get('device', self.device)
        hidden_dim = decodeur_params['hidden_dim']
        output_dim = decodeur_params['output_dim']
        bias1 = decodeur_params.get('bias1', True)
        bias2 = decodeur_params.get('bias2', True)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.end_channels, hidden_dim, bias=bias1),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim, bias=bias2)
        ).to(device)
        self._decoder_output_dim = output_dim

    def forward_horizon(self, y_prev=None, X_futur=None):
        if self.decoder is None:
            raise RuntimeError("Decoder has not been defined. Call define_horizon_decodeur first.")

        if y_prev is not None:
            decoder_input = y_prev
        elif X_futur is not None:
            decoder_input = X_futur
        elif self._decoder_input is not None:
            decoder_input = self._decoder_input
        else:
            raise RuntimeError("No input available for decoder. Provide y_prev or X_futur, or run a forward pass first.")

        return self.decoder(decoder_input)
        
class DilatedCNN(torch.nn.Module):
    def __init__(self, channels, dilations, lin_channels, end_channels, n_sequences, device, act_func, dropout, out_channels, task_type, use_layernorm=False, return_hidden=False, horizon=0):
        super(DilatedCNN, self).__init__()

        # Initialisation des listes pour les convolutions et les BatchNorm
        self.cnn_layer_list = []
        self.batch_norm_list = []
        self.num_layer = len(channels) - 1
        
        # Initialisation des couches convolutives et BatchNorm
        for i in range(self.num_layer):
            if i == 0:
                self.cnn_layer_list.append(torch.nn.Conv1d(channels[i] + end_channels if horizon > 0 else channels[i], channels[i + 1], kernel_size=3, padding='same', dilation=dilations[i], padding_mode='replicate').to(device))
            else:
                self.cnn_layer_list.append(torch.nn.Conv1d(channels[i], channels[i + 1], kernel_size=3, padding='same', dilation=dilations[i], padding_mode='replicate').to(device))
            if use_layernorm:
                self.batch_norm_list.append(torch.nn.LayerNorm(channels[i + 1]).to(device))
            else:
                self.batch_norm_list.append(torch.nn.BatchNorm1d(channels[i + 1]).to(device))

        self.dropout = torch.nn.Dropout(dropout)
        
        # Convertir les listes en ModuleList pour être compatible avec PyTorch
        self.cnn_layer_list = torch.nn.ModuleList(self.cnn_layer_list)
        self.batch_norm_list = torch.nn.ModuleList(self.batch_norm_list)
        
        # Dropout after GRU
        self.dropout = torch.nn.Dropout(p=dropout).to(device)

        # Output layer
        self.linear1 = torch.nn.Linear(channels[-1], lin_channels).to(device)
        self.linear2 = torch.nn.Linear(lin_channels, end_channels).to(device)
        self.output_layer = torch.nn.Linear(end_channels, out_channels).to(device)

        # Activation function
        self.act_func = getattr(torch.nn, act_func)()

        self.return_hidden = return_hidden
        self.device = device
        self.end_channels = end_channels
        self.decoder = None
        self._decoder_input = None
        self.horizon = horizon
        self.out_channels = out_channels
        self.n_sequences = n_sequences

        # Output activation depending on task
        if task_type == 'classification':
            self.output_activation = torch.nn.Softmax(dim=-1).to(device)
        elif task_type == 'binary':
            self.output_activation = torch.nn.Sigmoid().to(device)
        else:
            self.output_activation = torch.nn.Identity().to(device)  # For regression or custom handling

        if self.horizon > 0:
            self.define_horizon_decodeur()

    def forward(self, x, edges=None, z_prev=None):
        # Couche d'entrée

        if z_prev is None:
            z_prev = torch.zeros((x.shape[0], self.end_channels, self.n_sequences), device=x.device, dtype=x.dtype)
        else:
            z_prev = z_prev.view(x.shape[0], self.end_channels, self.n_sequences)
        
        if self.horizon > 0:
            x = torch.cat((x, z_prev), dim=1)

        # Couches convolutives dilatées avec BatchNorm, activation et dropout
        for cnn_layer, batch_norm in zip(self.cnn_layer_list, self.batch_norm_list):
            x = cnn_layer(x)
            x = batch_norm(x)  # Batch Normalization
            x = self.act_func(x)
            x = self.dropout(x)
        
        # Garder uniquement le dernier élément des séquences
        x = x[:, :, -1]

        # Activation and output
        #x = self.act_func(x)
        x = self.act_func(self.linear1(x))
        #x = self.dropout(x)
        hidden = self.act_func(self.linear2(x))
        #x = self.dropout(x)
        logits = self.output_layer(hidden)
        output = self.output_activation(logits)
        self._decoder_input = hidden
        return output, logits, hidden

    def define_horizon_decodeur(self, decodeur_params=None):
        if decodeur_params is None:
            if self.horizon <= 0:
                raise ValueError("horizon must be greater than zero to automatically define the decoder.")
            decodeur_params = {
                'device': self.device,
                'hidden_dim': self.end_channels,
                'output_dim': self.out_channels * self.horizon,
            }

        device = decodeur_params.get('device', self.device)
        hidden_dim = decodeur_params['hidden_dim']
        output_dim = decodeur_params['output_dim']
        bias1 = decodeur_params.get('bias1', True)
        bias2 = decodeur_params.get('bias2', True)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.end_channels, hidden_dim, bias=bias1),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim, bias=bias2)
        ).to(device)
        self._decoder_output_dim = output_dim

    def forward_horizon(self, y_prev=None, X_futur=None):
        if self.decoder is None:
            raise RuntimeError("Decoder has not been defined. Call define_horizon_decodeur first.")

        if y_prev is not None:
            decoder_input = y_prev
        elif X_futur is not None:
            decoder_input = X_futur
        elif self._decoder_input is not None:
            decoder_input = self._decoder_input
        else:
            raise RuntimeError("No input available for decoder. Provide y_prev or X_futur, or run a forward pass first.")

        return self.decoder(decoder_input)
        
