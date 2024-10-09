import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
from kan import KANLinear, KAN

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Encoder(nn.Module):
    def __init__(self, T, input_size, encoder_num_hidden, parallel=False):
        super(Encoder, self).__init__()
        self.encoder_num_hidden = encoder_num_hidden
        self.input_size = input_size
        self.parallel = parallel
        self.T = T

        self.S_A = KANLinear(input_size, input_size)

        self.KAN_input_A = KANLinear(2 * self.encoder_num_hidden + self.T,1)

        self.encoder_lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.encoder_num_hidden,
            num_layers=1
        )

        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=self.encoder_num_hidden, nhead=8,
                                    dim_feedforward=self.encoder_num_hidden * 4, dropout=0.7),
            num_layers=8)

        self.pos_encoder = PositionalEncoding(self.encoder_num_hidden, dropout=0.1)

    def forward(self, X):
        X_tilde = Variable(X.data.new(
            X.size(0), self.T, self.input_size).zero_())
        X_encoded = Variable(X.data.new(
            X.size(0), self.T, self.encoder_num_hidden).zero_())

        h_n = self._init_states(X)
        s_n = self._init_states(X)
        for t in range(self.T):
            x = torch.cat((h_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           s_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           X.permute(0, 2, 1)), dim=2)

            x = self.KAN_input_A(
                x.view(-1, self.encoder_num_hidden * 2 + self.T))

            alpha = F.softmax(x.view(-1, self.input_size), dim=1)
            x1 = X[:, t]
            alpha_t = F.sigmoid(self.S_A(x1))
            alpha1 = F.softmax(alpha_t, dim=1)

            x_tilde = torch.mul(alpha * alpha1, X[:, t, :])

            self.encoder_lstm.flatten_parameters()
            _, final_state = self.encoder_lstm(
                x_tilde.unsqueeze(0), (h_n, s_n))
            h_n = final_state[0]
            s_n = final_state[1]
            X_tilde[:, t, :] = x_tilde
            X_encoded[:, t, :] = h_n

        X_encoded = self.pos_encoder(X_encoded)
        X_encoded = self.transformer_encoder(X_encoded)
        return X_tilde, X_encoded

    def _init_states(self, X):
        return Variable(X.data.new(1, X.size(0), self.encoder_num_hidden).zero_())


class Decoder(nn.Module):
    def __init__(self, T, decoder_num_hidden, encoder_num_hidden, P):
        super(Decoder, self).__init__()
        self.decoder_num_hidden = decoder_num_hidden
        self.encoder_num_hidden = encoder_num_hidden
        self.T = T
        self.P = P

        self.attn_layer = nn.Sequential(
            nn.Linear(2 * decoder_num_hidden +
                      encoder_num_hidden, encoder_num_hidden),
            nn.Tanh(),
            nn.Linear(encoder_num_hidden, 1)
        )

        self.T_A = nn.Linear(self.T * self.encoder_num_hidden, self.T)

        self.lstm_layer = nn.LSTM(
            input_size=1,
            hidden_size=decoder_num_hidden
        )

        self.fc = nn.Linear(encoder_num_hidden + 1, 1)
        self.fc_final = nn.Linear(decoder_num_hidden + encoder_num_hidden, self.P)
        self.fc.weight.data.normal_()

    def forward(self, X_encoded, y_prev):
        d_n = self._init_states(X_encoded)
        c_n = self._init_states(X_encoded)
        for t in range(self.T):
            x = torch.cat((d_n.repeat(self.T, 1, 1).permute(1, 0, 2),
                           c_n.repeat(self.T, 1, 1).permute(1, 0, 2),
                           X_encoded), dim=2)
            beta = F.relu(self.attn_layer(
                x.view(-1, 2 * self.decoder_num_hidden + self.encoder_num_hidden)).view(-1, self.T))
            beta = F.softmax(beta, dim=1)
            context = torch.bmm(beta.unsqueeze(1), X_encoded)[:, 0, :]
            if t < self.T:
                y_tilde = self.fc(
                    torch.cat((context, y_prev[:, t].unsqueeze(1)), dim=1))
                self.lstm_layer.flatten_parameters()
                _, final_states = self.lstm_layer(
                    y_tilde.unsqueeze(0), (d_n, c_n))
                d_n = final_states[0]
                c_n = final_states[1]
        y_pred = self.fc_final(torch.cat((d_n[0], context), dim=1))
        y_pred = y_pred[:, -1:]
        return y_pred

    def _init_states(self, X):
        return Variable(X.data.new(1, X.size(0), self.decoder_num_hidden).zero_())


class EKLT(nn.Module):
    def __init__(self, X, y, T, P,
                 encoder_num_hidden,
                 decoder_num_hidden,
                 batch_size,
                 learning_rate,
                 epochs,
                 parallel=False):
        super(EKLT, self).__init__()
        self.encoder_num_hidden = encoder_num_hidden
        self.decoder_num_hidden = decoder_num_hidden
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.parallel = parallel
        self.shuffle = False
        self.epochs = epochs
        self.P = P
        self.T = T
        self.X = X
        self.y = y

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        print("==> Use accelerator: ", self.device)

        self.Encoder = Encoder(input_size=X.shape[1],
                               encoder_num_hidden=encoder_num_hidden,
                               T=T).to(self.device)
        self.Decoder = Decoder(encoder_num_hidden=encoder_num_hidden,
                               decoder_num_hidden=decoder_num_hidden,
                               T=T,
                               P=P).to(self.device)

        self.criterion = nn.MSELoss()

        if self.parallel:
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)

        self.encoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad,
                                                          self.Encoder.parameters()),
                                            lr=self.learning_rate)
        self.decoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad,
                                                          self.Decoder.parameters()),
                                            lr=self.learning_rate)

        self.train_timesteps = int(self.X.shape[0] * 0.8)
        self.input_size = self.X.shape[1]

    def train(self):
        iter_per_epoch = int(
            np.ceil(self.train_timesteps * 1. / self.batch_size))
        self.iter_losses = np.zeros(self.epochs * iter_per_epoch)
        self.epoch_losses = np.zeros(self.epochs)
        n_iter = 0
        for epoch in range(self.epochs):
            if self.shuffle:
                ref_idx = np.random.permutation(self.train_timesteps - self.T - self.P + 1)
            else:
                ref_idx = np.array(range(self.train_timesteps - self.T - self.P + 1))

            idx = 0

            while idx < self.train_timesteps:
                indices = ref_idx[idx:(idx + self.batch_size)]
                x = np.zeros((len(indices), self.T, self.input_size))
                y_prev = np.zeros((len(indices), self.T))
                y_gt = self.y[indices + self.T + self.P - 1]

                for bs in range(len(indices)):
                    x[bs, :, :] = self.X[indices[bs]:(
                            indices[bs] + self.T), :]
                    y_prev[bs, :] = self.y[indices[bs]: (indices[bs] + self.T)]

                loss = self.train_forward(x, y_prev, y_gt)
                self.iter_losses[int(
                    epoch * iter_per_epoch + idx / self.batch_size)] = loss

                idx += self.batch_size
                n_iter += 1

                if n_iter % 10000 == 0 and n_iter != 0:
                    for param_group in self.encoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9
                    for param_group in self.decoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9

                self.epoch_losses[epoch] = np.mean(self.iter_losses[range(
                    epoch * iter_per_epoch, (epoch + 1) * iter_per_epoch)])

            if epoch % 10 == 0:
                print("Epochs: ", epoch, " Iterations: ", n_iter,
                      " Loss: ", self.epoch_losses[epoch])

    def train_forward(self, X, y_prev, y_gt):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_weighted, input_encoded = self.Encoder(
            Variable(torch.from_numpy(X).type(torch.FloatTensor).to(self.device)))
        y_pred = self.Decoder(input_encoded, Variable(
            torch.from_numpy(y_prev).type(torch.FloatTensor).to(self.device)))

        y_true = Variable(torch.from_numpy(
            y_gt).type(torch.FloatTensor).to(self.device))

        y_true = y_true.view(-1, 1)
        loss = self.criterion(y_pred, y_true)
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item()

    def test(self, on_train=False):
        if on_train:
            y_pred = np.zeros(self.train_timesteps - self.T + 1 - self.P)
        else:
            y_pred = np.zeros(self.X.shape[0] - self.train_timesteps)

        i = 0
        while i < len(y_pred):
            batch_idx = np.array(range(len(y_pred)))[i: (i + self.batch_size)]
            X = np.zeros((len(batch_idx), self.T, self.X.shape[1]))
            y_history = np.zeros((len(batch_idx), self.T))

            for j in range(len(batch_idx)):
                if on_train:
                    X[j, :, :] = self.X[range(
                        batch_idx[j], batch_idx[j] + self.T), :]
                    y_history[j, :] = self.y[range(
                        batch_idx[j], batch_idx[j] + self.T)]
                else:
                    X[j, :, :] = self.X[range(
                        batch_idx[j] + self.train_timesteps - self.T, batch_idx[j] + self.train_timesteps), :]
                    y_history[j, :] = self.y[range(
                        batch_idx[j] + self.train_timesteps - self.T, batch_idx[j] + self.train_timesteps)]

            y_history = Variable(torch.from_numpy(
                y_history).type(torch.FloatTensor).to(self.device))
            _, input_encoded = self.Encoder(
                Variable(torch.from_numpy(X).type(torch.FloatTensor).to(self.device)))
            y_pred[i:(i + self.batch_size)] = self.Decoder(input_encoded,
                                                           y_history).cpu().data.numpy()[:, 0]
            i += self.batch_size

        return y_pred
