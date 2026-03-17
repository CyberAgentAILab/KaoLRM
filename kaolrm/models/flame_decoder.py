import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ChannelGating(nn.Module):
    def __init__(self, input_dim, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // reduction), nn.ReLU(), nn.Linear(input_dim // reduction, input_dim), nn.Sigmoid()
        )

    def forward(self, x):
        # x: [batch, input_dim]
        gate = self.fc(x)  # Learn gates per feature
        return x * gate  # Feature-wise gating


class AttentionMLPBlock(nn.Module):
    """
    Transformer-style block: Pre-LN self-attention followed by Pre-LN MLP.

    Pre-norm (LayerNorm before sublayer) is used for training stability.
    If input_dim != output_dim the attention branch acts as a projection layer
    and the residual shortcut is skipped for that sublayer.
    """

    def __init__(self, input_dim, output_dim, mlp_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(input_dim)
        self.ln2 = nn.LayerNorm(output_dim)
        self.qkv = nn.Linear(input_dim, output_dim * 3)
        self.attn_out = nn.Linear(output_dim, output_dim)
        self.mlp = nn.Sequential(nn.Linear(output_dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, output_dim))

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        _x = self.ln1(x)
        qkv = self.qkv(_x)
        q, k, v = qkv.chunk(3, dim=-1)
        # Scaled dot-product attention: divide by sqrt(d_k) to prevent vanishing gradients.
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = self.attn_out(attn_output)
        # Skip residual when dimensions differ (projection case).
        if x.shape[-1] == attn_output.shape[-1]:
            x2 = x + attn_output
        else:
            x2 = attn_output
        _x2 = self.ln2(x2)
        mlp_output = self.mlp(_x2)
        out = x2 + mlp_output
        return out


class FLAMEDecoder(nn.Module):
    """
    Decode a triplane feature map into FLAME 3DMM parameters.

    The triplane is flattened along the spatial dimensions, mean-pooled over
    the channel axis, then passed through an MLP to predict 160 parameters:

        Index range  | Parameter      | Dim | Description
        -------------|----------------|-----|----------------------------------
        [0,   100)   | shape          | 100 | FLAME shape coefficients (PCA)
        [100, 150)   | expression     |  50 | FLAME expression coefficients
        [150, 156)   | pose           |   6 | Jaw + neck pose (axis-angle ×2)
        [156, 157)   | scale          |   1 | Global mesh scale (Softplus > 0)
        [157, 160)   | translation    |   3 | Global XYZ translation
    """

    def __init__(self, triplane_dim, triplane_res):
        super().__init__()
        self.triplane_res = triplane_res
        self.triplane_dim = triplane_dim

        self.planes2seq = lambda x: rearrange(x, "B N C W H -> B (N W H) C", N=3, C=self.triplane_dim)
        seq_length = 3 * triplane_res * triplane_res

        self.decoder = nn.Sequential(
            nn.Linear(seq_length, 2048),
            nn.GELU(),
            nn.Linear(2048, 1024),
            nn.GELU(),
            ChannelGating(1024),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 160),
        )

        # Softplus guarantees scale > 0; clamped at 0.1 to avoid degenerate meshes.
        self.scale_act = nn.Softplus()

    def forward(self, planes):

        seq = self.planes2seq(planes)  # B, seq, channel
        seq = seq.mean(dim=2)  # B, seq
        x = self.decoder(seq)  # B, 160

        return {
            "shape": x[..., :100],
            "expression": x[..., 100:150],
            "pose": x[..., 150:156],
            "scale": self.scale_act(x[..., 156:157]).clamp(min=0.1),
            "translation": x[..., 157:160],
        }


if __name__ == "__main__":
    # --- Usage example ---
    batch, seq, input_dim, output_dim, mlp_dim = 8, 1000, 32, 1, 128
    model = AttentionMLPBlock(input_dim, output_dim, mlp_dim)

    x = torch.randn(batch, seq, input_dim)
    out = model(x)
    print(out.shape)  # [8, 10, 64]
