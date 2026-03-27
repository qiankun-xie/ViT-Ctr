# SimpViT模型 — 直接移植自ViT-RR, 仅修改num_outputs=3
import torch
import torch.nn as nn


class SimpViT(nn.Module):
    def __init__(self,
                 img_size=64,
                 patch_size=16,
                 num_outputs=3,
                 hidden_size=64,
                 num_layers=2,
                 num_heads=4):
        super(SimpViT, self).__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.patch_embedding = nn.Conv2d(in_channels=2, out_channels=hidden_size, kernel_size=patch_size, stride=patch_size)
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, hidden_size))
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, num_outputs)

    def forward(self, x):
        patches = self.patch_embedding(x)  # [batch_size, hidden_size, num_patches, num_patches]
        patches = patches.permute(0, 2, 3, 1)  # [batch_size, num_patches, num_patches, hidden_size]
        patches = patches.view(x.size(0), -1, patches.size(-1))  # [batch_size, num_patches*num_patches, hidden_size]

        patches = patches + self.position_embedding[:, :patches.size(1)]

        patches = patches.permute(1, 0, 2)  # [num_patches*num_patches, batch_size, hidden_size]
        encoded_patches = self.transformer_encoder(patches)

        class_token = encoded_patches.mean(dim=0)  # [batch_size, hidden_size]

        output = self.fc(class_token)
        return output
