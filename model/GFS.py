import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from utils.load_data import *
random_seed = 77
np.random.seed(random_seed)
torch.manual_seed(random_seed)

class GFS(nn.Module):
    def __init__(self, device, price_dim, trend_dim, feature_dim, seq):
        #price_dim = 10, trend_dim =7, feature_dim =..my choice (8)
        super().__init__()

        self.device = device

        #ALSTM for Price
        self.lstm_p = nn.LSTM(input_size = price_dim, hidden_size = feature_dim,
         num_layers=2, batch_first=True).to(device)
        self.attention_p = Attention(feature_dim=feature_dim).to(device)
   
        #ALSTM for global Trend
        self.lstm_t = nn.LSTM(input_size = trend_dim, hidden_size = feature_dim,
         num_layers=2, batch_first=True).to(device)
        self.attention_t = Attention(feature_dim=feature_dim).to(device)
        
        #GCN for Stationary graph
        self.conv1_p_1 = GCNConv(in_channels=feature_dim, out_channels=16, cached=False,
                             normalize=True).to(device)
        self.conv1_p_2 = GCNConv(in_channels=16, out_channels=feature_dim, cached=False,
                             normalize=True).to(device)

        # GCN for trend graph
        self.conv1_t_1 = GCNConv(in_channels=feature_dim, out_channels=16, cached=False,
                             normalize=True).to(device)
        self.conv1_t_2 = GCNConv(in_channels=16, out_channels=feature_dim, cached=False,
                             normalize=True).to(device)

        #MLP
        self.mlp_fc1 = nn.Linear(feature_dim + 5, 1).to(device)
        # self.mlp_fc1 = nn.Linear(feature_dim + 5, 8).to(device)
        # self.mlp_fc2 = nn.Linear(8, 4).to(device)
        # self.mlp_fc3 = nn.Linear(4, 1).to(device)


    def forward(self, price, global_trend, price_graph, trend_graph):
        
        #extract price feature representations
        x, _ = self.lstm_p(price)
        x = x.squeeze()
        x, _ = self.attention_p(x)  
       

        #extract global trend feature representations
        g, _ = self.lstm_t(global_trend)
        g = g.squeeze()
        g, __ = self.attention_t(g)
        # print(g.shape)
        
        #set up price graph (stationary)
        p_edge_index, p_weights, p_ = price_graph[0], price_graph[1], price_graph[2]
      

        #set up trend graph (changes everytime)
        t_edge_index, t_weights, t_ = trend_graph[0], trend_graph[1], trend_graph[2]
        
        p_edge_index= p_edge_index.to(self.device)
        p_weights = p_weights.to(self.device)
        t_edge_index = torch.LongTensor(t_edge_index).to(self.device)
        t_weights = torch.tensor(t_weights, dtype=torch.float32).to(self.device)

        # Price graph embeddings
        PG_EMB = self.conv1_p_1(x, p_edge_index, p_weights)
        PG_EMB = self.conv1_p_2(PG_EMB, p_edge_index, p_weights)
        
        # Trend graph embeddings
        TG_EMB = self.conv1_t_1(x, t_edge_index, t_weights)
        TG_EMB = self.conv1_t_2(TG_EMB, t_edge_index, t_weights)
        
        # print(PG_EMB.shape)
        # print(TG_EMB.shape)
        # print(g.shape)

        g = g.transpose(1,0)
        
        trend_EMB = torch.matmul(TG_EMB, g)
        # print(trend_EMB)

        
        final_EMB = torch.cat([PG_EMB, trend_EMB], dim=1)
        
        # out_ = self.mlp_fc3(self.mlp_fc2(self.mlp_fc1(final_EMB)))
        out_ =self.mlp_fc1(final_EMB)
        out = torch.sigmoid(out_)
        # print(out)
           
        return out


class Attention(nn.Module):
    def __init__(self, feature_dim):
        super(Attention, self).__init__()

        self.attn_1 = nn.Linear(feature_dim, feature_dim)
        self.attn_2 = nn.Linear(feature_dim, 1)

        # inititalize
        nn.init.xavier_uniform_(self.attn_1.weight)
        nn.init.xavier_uniform_(self.attn_2.weight)
        self.attn_1.bias.data.fill_(0.0)
        self.attn_2.bias.data.fill_(0.0)

    def forward(self, x, return_attention=False):
        """
        Input x is encoder output
        return_attention decides whether to return
        attention scores over the encoder output
        """
        sequence_length = x.shape[1]

        self_attention_scores = self.attn_2(torch.tanh(self.attn_1(x)))

        # Attend for each time step using the previous context
        context_vectors = []
        attention_vectors = []

        for t in range(sequence_length):
            # For each timestep the context that is attented grows
            # as there are more available previous hidden states
            weighted_attention_scores = F.softmax(
                self_attention_scores[:, :t + 1, :].clone(), dim=1)

            context_vectors.append(
                torch.sum(weighted_attention_scores * x[:, :t + 1, :].clone(), dim=1))

            if return_attention:
                attention_vectors.append(
                    weighted_attention_scores.cpu().detach().numpy())

        context_vectors = torch.stack(context_vectors).transpose(0, 1)
        context_vectors = torch.sum(context_vectors, dim=1)
        # print(context_vectors)
        # print(context_vectors.shape)
        return context_vectors, attention_vectors