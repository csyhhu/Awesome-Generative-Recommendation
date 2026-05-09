import torch
import torch.nn as nn
import torch.nn.functional as F


class UniMixingLite(nn.Module):
    """
    UniMixing-Lite Block (单层)。

    Args:
        L         : 输入 embedding 总维度
        B         : block size，L 必须能被 B 整除，N = L // B 为 block 数量
        b         : 基矩阵数量，论文消融最优 b=4
        r         : 全局矩阵低秩秩数，论文消融最优 r=128
        n_sinkhorn: Sinkhorn-Knopp 迭代次数，默认 5
        tau       : 温度系数，越小越接近置换矩阵，论文最终用 0.05
    """

    def __init__(self, L: int, B: int, b: int, r: int, n_sinkhorn: int = 5, tau: float = 0.05):
        super().__init__()
        assert L % B == 0, f"L={L} must be divisible by B={B}"
        self.L = L
        self.B = B
        self.b = b
        self.r = r
        self.n_sinkhorn = n_sinkhorn
        self.tau = tau
        self.N = L // B

        # ── 局部混合参数 ────────────────────────────────────────────────────
        # 共享基矩阵 Z_l in R^{B x B}，共 b 个；shape: (b, B, B)
        self.Z = nn.Parameter(torch.randn(b, B, B) * 0.02)
        # 每个 block 的组合系数 omega^i in R^b，共 N 个；shape: (N, b)
        self.omega = nn.Parameter(torch.randn(self.N, b) * 0.02)

        # ── 全局混合参数（低秩分解）──────────────────────────────────────────
        # W_G ≈ A_G @ B_G
        self.A_G = nn.Parameter(torch.randn(self.N, r) * 0.02)  # (N, r)
        self.B_G = nn.Parameter(torch.randn(r, self.N) * 0.02)  # (r, N)

        # ── 归一化层 ──────────────────────────────────────────────────────────
        self.norm = nn.RMSNorm(L)

    def sinkhorn(self, M: torch.Tensor) -> torch.Tensor:
        """
        Sinkhorn-Knopp 迭代：将任意正矩阵归一化为双随机矩阵。
        流程：对称化 -> 温度缩放 -> exp -> 交替行列归一化。

        Args:
            M: (..., n, n)  最后两维做归一化，支持 batch 维度
        Returns:
            双随机矩阵，shape 与输入相同
        """
        # Step 1: 对称化，满足论文对称性约束 (W + W^T) / 2
        M = (M + M.transpose(-1, -2)) / 2          # (..., n, n)
        # Step 2: 除以温度再 exp，保证所有元素为正（Sinkhorn 的前提）
        M = torch.exp(M / self.tau)                 # (..., n, n)
        # Step 3: 交替行列归一化
        for _ in range(self.n_sinkhorn):
            M = M / M.sum(dim=-1, keepdim=True)     # 行归一化：每行和 = 1
            M = M / M.sum(dim=-2, keepdim=True)     # 列归一化：每列和 = 1
        return M                                    # (..., n, n) 双随机矩阵

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        前向传播，对应论文 Eq.8。
        Args:
            X      : (batch, L)  展平的 embedding 输入
        Returns:
            output : (batch, L)  UniMixing-Lite + 残差归一化后的输出
        """
        batch = X.shape[0]
        N, B = self.N, self.B

        # Step 1: 构造局部混合矩阵 W_B_star (N, B, B)
        # W_B^{*i} = Sinkhorn( sum_{l=1}^{b} omega^i_l * Z_l )
        # omega: (N, b) -> softmax 归一化组合系数
        omega_norm = F.softmax(self.omega, dim=-1)              # (N, b)
        # 加权求和基矩阵: (N, b) x (b, B, B) -> (N, B, B)
        W_B_raw = torch.einsum("nb,bxy->nxy", omega_norm, self.Z)  # (N, B, B)
        # Sinkhorn 约束，批量处理 N 个 block
        W_B_star = self.sinkhorn(W_B_raw)                       # (N, B, B)

        # Step 2: 构造全局混合矩阵 W_r (N, N)
        # W_r = Sinkhorn( A_G @ B_G )
        W_G_raw = self.A_G @ self.B_G                           # (N, N) 低秩近似
        W_r = self.sinkhorn(W_G_raw)                            # (N, N) 双随机全局矩阵

        # Step 3: 局部混合  H_i = x_i @ W_B_star_i
        # 类比 Heterogeneous Attention 中的 Value 投影
        X_blocks = X.view(batch, N, B)                          # (batch, N, B)
        # (batch, N, B) x (N, B, B) -> (batch, N, B)
        H = torch.einsum("bnx,nxy->bny", X_blocks, W_B_star)    # (batch, N, B)

        # Step 4: 全局混合  out = W_r @ H
        # 类比 Attention 中的 attention_score @ V
        # W_r: (N, N)  x  H: (batch, N, B)  ->  out: (batch, N, B)
        out = torch.einsum("mn,bnB->bmB", W_r, H)               # (batch, N, B)
        out = out.reshape(batch, self.L)                        # (batch, L)

        # Step 5: 残差连接 + RMSNorm
        # O = RMSNorm(X + UniMixing-Lite(X))
        output = self.norm(X + out)                             # (batch, L)
        return output


# ─────────────────────────────────────────────────────────────────────────────
# 使用示例 & 维度验证
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 论文实验设置：L=768, B=6 -> N=128 个 block
    L = 768   # embedding 总维度
    B = 6     # block size
    b = 4     # 基矩阵数量（论文消融最优）
    r = 128   # 全局低秩秩数（论文消融最优）

    model = UniMixingLite(L=L, B=B, b=b, r=r, n_sinkhorn=5, tau=0.05)

    batch_size = 32
    x = torch.randn(batch_size, L)    # (32, 768)
    out = model(x)

    print(f"输入  X    : {x.shape}")
    print(f"输出  out  : {out.shape}")

    # 参数量分析
    total = sum(p.numel() for p in model.parameters())
    N = L // B
    print(); ___=None; print(f"参数量明细（N={N}, B={B}, b={b}, r={r}）：")
    print(f"  Z  (基矩阵)  b x B x B = {b} x {B} x {B} = {b*B*B}")
    print(f"  omega        N x b     = {N} x {b}     = {N*b}")
    print(f"  A_G          N x r     = {N} x {r}   = {N*r}")
    print(f"  B_G          r x N     = {r} x {N}   = {r*N}")
    print(f"  RMSNorm      L         = {L}")
    print(f"  总参数量               = {total:,}")

    # 与 UniMixing（非 Lite）参数量对比
    unimixing_local  = N * B * B   # N 个独立 W_B^i
    unimixing_global = N * N       # 完整 W_G
    print(); ___=None; print(f"对比 UniMixing（非 Lite）：")
    print(f"  局部  N x B x B = {N} x {B} x {B} = {unimixing_local:,}")
    print(f"  全局  N x N     = {N} x {N}   = {unimixing_global:,}")
    print(f"  合计            = {unimixing_local + unimixing_global:,}")
    print(f"  参数压缩比      = {(unimixing_local + unimixing_global) / total:.1f}x")
