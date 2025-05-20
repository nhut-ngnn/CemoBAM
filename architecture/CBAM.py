import torch
import torch.nn as nn
import torch.nn.functional as F

class CBAM1D(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention_mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=False) 
        max_pool, _ = torch.max(x, dim=1, keepdim=False) 

        avg_pool = x
        max_pool = x

        channel_attn = self.channel_attention_mlp(avg_pool) + self.channel_attention_mlp(max_pool)
        x = x * channel_attn 

        x_ = x.unsqueeze(1)  
        avg_pool = torch.mean(x_, dim=1, keepdim=True)  
        max_pool, _ = torch.max(x_, dim=1, keepdim=True) 

        spatial_input = torch.cat([avg_pool, max_pool], dim=1)  
        spatial_attn = self.spatial_attention(spatial_input) 

        x = x * spatial_attn.squeeze(1)  

        return x


class CrossModalFusion(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.3, fusion_head_output_type="min"):
        super().__init__()

        self.text_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.text_linear = nn.Linear(hidden_dim, hidden_dim)
        self.text_layer_norm = nn.LayerNorm(hidden_dim)

        self.audio_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.audio_linear = nn.Linear(hidden_dim, hidden_dim)
        self.audio_layer_norm = nn.LayerNorm(hidden_dim)

        self.cbam_text = CBAM1D(hidden_dim)
        self.cbam_audio = CBAM1D(hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.fusion_head_output_type = fusion_head_output_type

    def forward(self, text_feat, audio_feat): 
        text_feat = text_feat.unsqueeze(1)    
        audio_feat = audio_feat.unsqueeze(1) 

        audio_attn, _ = self.audio_attention(query=text_feat, key=audio_feat, value=audio_feat)
        audio_attn = self.audio_layer_norm(self.audio_linear(audio_attn) + text_feat)
        audio_attn = self.dropout(audio_attn).squeeze(1)  
        audio_attn = self.cbam_audio(audio_attn)

        text_attn, _ = self.text_attention(query=audio_feat, key=text_feat, value=text_feat)
        text_attn = self.text_layer_norm(self.text_linear(text_attn) + audio_feat)
        text_attn = self.dropout(text_attn).squeeze(1)  
        text_attn = self.cbam_text(text_attn)

        fusion = torch.stack((text_attn, audio_attn), dim=1)

        if self.fusion_head_output_type == "cls":
            fused = fusion[:, 0, :]  
        elif self.fusion_head_output_type == "mean":
            fused = fusion.mean(dim=1)
        elif self.fusion_head_output_type == "max":
            fused = fusion.max(dim=1)[0]
        elif self.fusion_head_output_type == "min":
            fused = fusion.min(dim=1)[0]
        elif self.fusion_head_output_type == "concat":
            fused = torch.cat([text_attn, audio_attn], dim=1)
        else:
            raise ValueError(f"Invalid fusion type: {self.fusion_head_output_type}")

        return fused 
