import torch
import torch.nn as nn
from architecture.GAT_module import GATLayers
from architecture.CBAM import CrossModalFusion

class MultiModalGNN(nn.Module):
    def __init__(self, hidden_dim=512, num_classes=4, dropout=0.3, heads=4, num_layers=3, fusion_head_output_type='min'):
        super(MultiModalGNN, self).__init__()

        self.text_projection = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.audio_projection = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.cross_fusion = CrossModalFusion(hidden_dim, num_heads=heads, dropout=dropout, fusion_head_output_type=fusion_head_output_type)
        self.gnn = GATLayers(hidden_dim, heads=heads, num_layers=num_layers, dropout=dropout)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, text_x, audio_x, edge_index):
        text_feat = self.text_projection(text_x)
        audio_feat = self.audio_projection(audio_x)

        gnn_input = torch.cat([text_feat, audio_feat], dim=1)
        graph_feat = self.gnn(gnn_input, edge_index)

        CMT_feature = self.cross_fusion(text_feat, audio_feat)

        combined = torch.cat([graph_feat, CMT_feature], dim=1)
        out = self.mlp(combined)
        return out

    def get_fused_embeddings(self, text_x, audio_x):
        text_feat = self.text_projection(text_x)
        audio_feat = self.audio_projection(audio_x)
        CMT_feature = self.cross_fusion(text_feat, audio_feat)
        return CMT_feature
