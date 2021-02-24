import torch
import torch.nn as nn
from area_attention import AreaAttention, MultiHeadAreaAttention
import json
from typing import List, Tuple
from einops import rearrange, repeat

MIN_NUM_PATCHES = 16

############### Define AreaAttention Vit class########################


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class AreaAttentionWrapper(nn.Module):
    def __init__(self, dim, dim_head, max_area_height, max_area_width,
                 memory_height, memory_width, heads=8, dropout=0.):
        super().__init__()

        self.to_qkv = nn.Linear(dim, 3 * dim, bias=False)

        area_attention_head = AreaAttention(key_query_size=dim_head,
                                            max_area_height=max_area_height,
                                            max_area_width=max_area_width,
                                            memory_height=memory_height,
                                            memory_width=memory_width,
                                            dropout_rate=dropout)
        self.multihead_area_attention = MultiHeadAreaAttention(area_attention=area_attention_head,
                                                               num_heads=heads,
                                                               key_query_size=dim,
                                                               key_query_size_hidden=dim_head,
                                                               value_size=dim,
                                                               value_size_hidden=dim_head)

    def forward(self, x, mask=None):
        assert mask is None, 'Mask is currently not supported for area_attention'
        # x : [batch, num_tokens, dim]
        qkv = self.to_qkv(x)  # [batch, num_tokens, 3 * dim]
        q, k, v = rearrange(qkv, 'b n (qkv d) -> qkv b n d', qkv=3)  # 3 vectors of [batch, num_tokens, dim]

        return self.multihead_area_attention(q, k, v)

class Transformer(nn.Module):

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, aa_sizes: List[Tuple] = None, aa_memory_dims: Tuple = None):
        super().__init__()
        self.layers = nn.ModuleList([])

        if aa_sizes is None:
            aa_sizes = []

        for _, layer_indx in enumerate(range(depth)):
            if layer_indx + 1 <= len(aa_sizes):
                assert aa_memory_dims is not None, 'aa_sizes is not empty but memory dims were not specified'
                aa_h, aa_w = aa_sizes[layer_indx]
                self.layers.append(nn.ModuleList([
                    Residual(PreNorm(dim, AreaAttentionWrapper(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout,
                                                                max_area_height=aa_h, max_area_width=aa_w,
                                                                memory_height=aa_memory_dims[0], memory_width=aa_memory_dims[1]))),
                    Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
                ]))
            else:
                self.layers.append(nn.ModuleList([
                    Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                    Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
                ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
                 pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.,
                 aa_sizes: List[Tuple] = None, aa_memory_dims: Tuple = None):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert num_patches >= MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, aa_sizes, aa_memory_dims)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img, mask=None):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x, mask)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
######################################################################


def get_aa_vit(cfg_file_path: str) -> ViT:
    """
    Method builds an AA ViT based on the cfg file and returns it
    @param cfg_file_path : path to the configuration file
    """

    with open(cfg_file_path) as f:
        cfg = json.load(f)

    print('Creating AA ViT using the following config:')
    print(json.dumps(cfg, indent=4))

    return ViT(image_size=cfg['image_size'], patch_size=cfg['patch_size'], num_classes=cfg['num_classes'],
               channels=cfg['num_channels'], dim=cfg['embedding_dim'], dim_head=cfg['head_dim'],
                depth=cfg['depth'], heads=cfg['num_heads'], mlp_dim=cfg['mlp_dim'], aa_memory_dims=cfg['aa_memory_dim'],
                aa_sizes=cfg['aa_sizes'])
