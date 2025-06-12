import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int=3, embedding_dim: int=256, patch_size: int=32):
        super().__init__()
        
        self.patcher = nn.Conv2d(in_channels=in_channels, out_channels=embedding_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.patcher(x)
        x = x.flatten(2).transpose(1, 2)
        return x
    
class MSABlock(nn.Module):
    def __init__(self, embedding_dim: int=256, num_heads: int=4, attn_dropout: float=0.0):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.msa = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=attn_dropout)
        
    def forward(self, x):
        x = self.layer_norm(x)
        output, _ = self.msa(query=x, key=x, value=x, need_weights=False)
        return output
        
class MLPBlock(nn.Module):
    def __init__(self, embedding_dim: int=256, mlp_size: int=512, dropout: float=0.1):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size, out_features=embedding_dim),
            nn.Dropout(p=0.5)
        )
        
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x
    
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim: int=256, num_heads: int=4, mlp_size: int=512, mlp_dropout: float=0.1, attn_dropout: float=0.0):
        super().__init__()
        
        self.msa_block = MSABlock(embedding_dim=embedding_dim, num_heads=num_heads, attn_dropout=attn_dropout)
        self.mlp_block = MLPBlock(embedding_dim=embedding_dim, mlp_size=mlp_size, dropout=mlp_dropout)
        
    def forward(self, x):
        x = x + self.msa_block(x)
        x = x + self.mlp_block(x)
        return x
    
class ViT(nn.Module):
    def __init__(self, img_size: int=224, in_channels: int=3,
                 patch_size: int=32, num_heads: int=4, num_transformer_layers: int=3,
                 embedding_dim: int=256, mlp_size: int=512, attn_dropout: float=0.0,
                 mlp_dropout: float=0.1, embedding_dropout: float=0.1, num_classes: int=2):
        super().__init__()
        
        self.num_patches = (img_size * img_size) // patch_size ** 2
        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim), requires_grad=True)
        self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches + 1, embedding_dim), requires_grad=True)
        
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.patch_embedding = PatchEmbedding(in_channels=in_channels, embedding_dim=embedding_dim, patch_size=patch_size)
        
        self.transformer_encoder = nn.Sequential(
            *[TransformerEncoderBlock(embedding_dim=embedding_dim, num_heads=num_heads, mlp_size=mlp_size, mlp_dropout=mlp_dropout, attn_dropout=attn_dropout)
            for _ in range(num_transformer_layers)]
        )
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=num_classes)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        class_token = self.class_embedding.expand(batch_size, -1, -1)
        x = self.patch_embedding(x)
        x = torch.cat((class_token, x), dim=1)
        x += self.position_embedding
        x = self.embedding_dropout(x)
        x = self.transformer_encoder(x)
        x = self.classifier(x[:, 0])
        return x
    
class CustomViTFusionModel(nn.Module):
    def __init__(self, num_classes: int=2):
        super().__init__()
        
        self.vit = ViT(num_classes=2).to(device)
        self.vit.classifier = nn.Identity()
        
        self.physio_net = nn.Sequential(
            nn.Linear(in_features=3, out_features=32),
            nn.ReLU(),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=256 + 32, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=64, out_features=num_classes)
        )
        
    def forward(self, image, physio_vec):
        x_img = self.vit(image)
        x_physio = self.physio_net(physio_vec)
        
        x = torch.cat((x_img, x_physio), dim=1)
        return self.classifier(x)