import torch
from .transformer_encoder import Transformer
from .TransCNN import HARTransformer, Gaussian_Position
import torch.nn.functional as F
from einops import rearrange

class HARTrans(torch.nn.Module):
    def __init__(self, K, sample, n_seq, input_dim, d_model, n_class, hlayers, vlayers, hheads, vheads):
        super(HARTrans, self).__init__()
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.linear_input = torch.nn.Linear(input_dim//sample, d_model)
        self.transformer = HARTransformer(d_model, hlayers, hheads, n_seq)
        self.sample = sample
        self.n_seq = n_seq
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_class = n_class
        self.kernel_num = 128
        self.kernel_num_v = 16
        self.filter_sizes = [10, 40]
        self.filter_sizes_v = [2, 4]
        self.pos_encoding = Gaussian_Position(d_model, n_seq, K)

        if vlayers == 0:
            self.v_transformer = None
            self.dense = torch.nn.Linear(d_model, n_class)
        else:
            self.v_transformer = Transformer(n_seq, vlayers, vheads)
            self.dense = torch.nn.Linear(self.kernel_num * len(self.filter_sizes) + self.kernel_num_v * len(self.filter_sizes_v), n_class)

        self.dense2 = torch.nn.Linear(self.kernel_num * len(self.filter_sizes), 7)
        self.dropout_rate = 0.5
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.encoders = []
        self.encoder_v = []
        for i, filter_size in enumerate(self.filter_sizes):
            enc_attr_name = "encoder_%d" % i
            self.__setattr__(enc_attr_name,
                             torch.nn.Conv1d(in_channels=d_model,
                                       out_channels=self.kernel_num,
                                       kernel_size=filter_size)
                             )
            self.encoders.append(self.__getattr__(enc_attr_name))
        for i, filter_size in enumerate(self.filter_sizes_v):
            enc_attr_name_v = "encoder_v_%d" % i
            self.__setattr__(enc_attr_name_v,
                             torch.nn.Conv1d(in_channels=n_seq,
                                       out_channels=self.kernel_num_v,
                                       kernel_size=filter_size)
                             )
            self.encoder_v.append(self.__getattr__(enc_attr_name_v))

    def _aggregate(self, o, v=None):
        enc_outs = []
        enc_outs_v = []
        for encoder in self.encoders:
            f_map = encoder(o.transpose(-1, -2))
            enc_ = F.relu(f_map)
            k_h = enc_.size()[-1]
            enc_ = F.max_pool1d(enc_, kernel_size=k_h)
            enc_ = enc_.squeeze(dim=-1)
            enc_outs.append(enc_)
        encoding = self.dropout(torch.cat(enc_outs, 1))
        q_re = F.relu(encoding)
        if self.v_transformer is not None:
            for encoder in self.encoder_v:
                f_map = encoder(v.transpose(-1, -2))
                enc_ = F.relu(f_map)
                k_h = enc_.size()[-1]
                enc_ = F.max_pool1d(enc_, kernel_size=k_h)
                enc_ = enc_.squeeze(dim=-1)
                enc_outs_v.append(enc_)
            encoding_v = self.dropout(torch.cat(enc_outs_v, 1))
            v_re = F.relu(encoding_v)
            q_re = torch.cat((q_re, v_re), dim=1)
        return q_re

    def forward(self, data):
        #[b, s, d]
        
        # d1 = data.size(dim=0)
        # d3 = data.size(2)
        # x = data.unsqueeze(-2)
        # x = data.view(d1, -1, self.sample, d3)
        # x = torch.sum(x, dim=-2).squeeze(-2)
        # x = torch.div(x, self.sample)
        # x = self.pos_encoding(x)
        
        x = rearrange(data, 'b s (d spl) -> b s d spl', spl=self.sample)
        x = torch.sum(x, dim=-1)
        x = torch.div(x, self.sample)
        x = self.linear_input(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)

        if self.v_transformer is not None:
            # y = data.view(-1, self.n_seq, 1, self.d_model)
            y = rearrange(data, 'b s (d spl) -> b s d spl', spl=1)
            y = torch.sum(y, dim=-1)
            y = y.transpose(-1, -2)
            y = self.v_transformer(y)
            re = self._aggregate(x, y)
            predict = self.softmax(self.dense(re))
        else:
            re = self._aggregate(x)
            predict = self.softmax(self.dense2(re))

        return predict


class TransCNN(torch.nn.Module):
    def __init__(self, args):
        super(TransCNN, self).__init__()
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.transformer = Transformer(90, args.hlayers, 9)
        self.args = args
        self.kernel_num = 128
        self.kernel_num_v = 16
        self.filter_sizes = [10, 40]
        self.filter_sizes_v = [2, 4]

        if args.vlayers == 0:
            self.v_transformer = None
            self.dense = torch.nn.Linear(90, 7)
        else:
            self.v_transformer = Transformer(2000, args.vlayers, 200)
            self.dense = torch.nn.Linear(self.kernel_num * len(self.filter_sizes) + self.kernel_num_v * len(self.filter_sizes_v), 7)

        self.dense2 = torch.nn.Linear(self.kernel_num * len(self.filter_sizes), 7)
        self.dropout_rate = 0.5
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.encoders = []
        self.encoder_v = []
        for i, filter_size in enumerate(self.filter_sizes):
            enc_attr_name = "encoder_%d" % i
            self.__setattr__(enc_attr_name,
                             torch.nn.Conv1d(in_channels=90,
                                       out_channels=self.kernel_num,
                                       kernel_size=filter_size).to('cuda')
                             )
            self.encoders.append(self.__getattr__(enc_attr_name))
        for i, filter_size in enumerate(self.filter_sizes_v):
            enc_attr_name_v = "encoder_v_%d" % i
            self.__setattr__(enc_attr_name_v,
                             torch.nn.Conv1d(in_channels=2000,
                                       out_channels=self.kernel_num_v,
                                       kernel_size=filter_size).to('cuda')
                             )
            self.encoder_v.append(self.__getattr__(enc_attr_name_v))

    def _aggregate(self, o, v=None):
        enc_outs = []
        enc_outs_v = []
        for encoder in self.encoders:
            f_map = encoder(o.transpose(-1, -2))
            enc_ = F.relu(f_map)
            k_h = enc_.size()[-1]
            enc_ = F.max_pool1d(enc_, kernel_size=k_h)
            enc_ = enc_.squeeze(dim=-1)
            enc_outs.append(enc_)
        encoding = self.dropout(torch.cat(enc_outs, 1))
        q_re = F.relu(encoding)
        if self.v_transformer is not None:
            for encoder in self.encoder_v:
                f_map = encoder(v.transpose(-1, -2))
                enc_ = F.relu(f_map)
                k_h = enc_.size()[-1]
                enc_ = F.max_pool1d(enc_, kernel_size=k_h)
                enc_ = enc_.squeeze(dim=-1)
                enc_outs_v.append(enc_)
            encoding_v = self.dropout(torch.cat(enc_outs_v, 1))
            v_re = F.relu(encoding_v)
            q_re = torch.cat((q_re, v_re), dim=1)
        return q_re

    def forward(self, data):
        d1 = data.size(dim=0)
        d3 = data.size(2)
        x = data.unsqueeze(-2)
        x = data.view(d1, -1, self.args.sample, d3)
        
        
        x = torch.sum(x, dim=-2).squeeze(-2)
        x = torch.div(x, self.args.sample)
        x = self.transformer(x)

        if self.v_transformer is not None:
            y = data.view(-1, 2000, 3, 30)
            y = torch.sum(y, dim=-2).squeeze(-2)
            y = y.transpose(-1, -2)
            y = self.v_transformer(y)
            re = self._aggregate(x, y)
            predict = self.softmax(self.dense(re))
        else:
            re = self._aggregate(x)
            predict = self.softmax(self.dense2(re))

        return predict


class TransformerM(torch.nn.Module):
    def __init__(self, args):
        super(TransformerM, self).__init__()
        self.args = args
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.transformer = Transformer(90, args.hlayers, 9)

        if args.vlayers == 0:
            self.v_transformer = None
            self.dense = torch.nn.Linear(90, 7)
        else:
            self.v_transformer = Transformer(2000, args.vlayers, 200)
            self.linear = torch.nn.Linear(2000, 7)
            self.dense = torch.nn.Linear(90, 7)

        #self.linear = torch.nn.Linear(2000, args.com_dim)
        self.cls = torch.nn.Parameter(torch.zeros([1, 1, 90], dtype=torch.float, requires_grad=True))
        self.sep = torch.nn.Parameter(torch.zeros([1, 1, 90], dtype=torch.float, requires_grad=True))
        torch.nn.init.xavier_uniform_(self.cls, gain=1)
        torch.nn.init.xavier_uniform_(self.sep, gain=1)

    def fusion(self, x, y):
        y = self.softmax(self.linear(y))
        x = self.softmax(self.dense(x))
        predict = x + y
        return predict

    def forward(self, data):
        d1 = data.size(dim=0)
        d3 = data.size(2)
        x = data.unsqueeze(-2)
        x = data.view(d1, -1, self.args.sample, d3)
        x = torch.sum(x, dim=-2).squeeze(-2)
        x = torch.div(x, self.args.sample)
        #x = torch.cat((self.cls.repeat(d1, 1, 1), x), dim=1)
        dx = x.size(1)
        x = self.transformer(x)
        x = torch.div(torch.sum(x, dim=1).squeeze(dim=1), dx)
        #x = x[:, 0, :]
        if self.v_transformer is not None:
            y = data.view(-1, 2000, 3, 30)
            y = torch.sum(y, dim=-2).squeeze(-2)
            y = y.transpose(-1, -2)
            d2 = y.size(1)
            y = self.v_transformer(y)
            dy = y.size(1)*3
            y = torch.div(torch.sum(y, dim=1).squeeze(dim=1), dy)
            predict = self.fusion(x, y)
        else:
            predict = self.softmax(self.dense(x))

        return predict