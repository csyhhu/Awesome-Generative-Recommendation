"""
UniMixing-Lite: a lightweight unified feature mixing module for recommender systems.

Reference:
    Ha et al., "UniMixer: A Unified Architecture for Scaling Laws in
    Recommendation Systems", NeurIPS 2024. arXiv:2604.00590.

This module implements the UniMixing-Lite block (Eq. 4.3.8 of the paper) along
with a `reduce_method` switch that degenerates the module into one of the three
baseline methods (self-attention / TokenMixer / Wukong-FM) under the unified
theoretical framework (Eq. 4.3.7). This is useful for visualizing how the
learned W_G (global mixing) and W_B (local mixing) differ across paradigms.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# A minimal RMSNorm implementation so the script does not depend on having
# torch.nn.RMSNorm (only available since PyTorch 2.4). Drop-in compatible.
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Root Mean Square Normalization (drop-in replacement for nn.RMSNorm)."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))

    def forward(self, x):
        # Compute RMS over the last len(normalized_shape) dims.
        dims = tuple(range(-len(self.normalized_shape), 0))
        rms = torch.rsqrt(x.pow(2).mean(dim=dims, keepdim=True) + self.eps)
        return self.weight * x * rms


class UniMixingLite(nn.Module):
    """
    UniMixing-Lite Block (single layer).

    When `reduce_method is None`, this is the full UniMixing-Lite module:
        * Local mixing  W_B^{*i} = Sinkhorn-Knopp(sum_l omega_l^i * Z_l)
          (shared basis matrices Z_l combined per-block by omega^i)
        * Global mixing W_r       = Sinkhorn-Knopp(A_G @ B_G)
          (low-rank approximation of W_G)

    When `reduce_method` is set, the module degenerates into the corresponding
    baseline under the unified framework (Eq. 4.3.7 + Tab_0301):
        * 'self_attention': Local = X W_V (shared),  Global = softmax(QK^T / sqrt(d))
        * 'tokenmixer'    : Local = X (identity),    Global = fixed permutation matrix
        * 'wukong'        : Local = Y (fixed),        Global = X X^T (input-dependent)
        * 'kda'           : Local = X W_V (shared),  Global = G^KDA(X) (cumulative
                                                                 path-dependent,
                                                                 lower-triangular)

    Args:
        L              : total input embedding dimension (L = T * D in the paper).
        block_size     : size of each local mixing block (B in the paper); L % B == 0.
        n_basis        : number of shared basis matrices for local mixing (b in the paper).
        r              : rank for low-rank approximation of the global matrix W_G.
        n_sinkhorn     : number of Sinkhorn-Knopp iterations.
        tau            : temperature coefficient; smaller -> closer to a permutation matrix.
        reduce_method  : one of {None, 'self_attention', 'tokenmixer', 'wukong'}.
    """

    REDUCE_METHODS = (None, "self_attention", "tokenmixer", "wukong", "kda")

    def __init__(
        self,
        L: int,
        block_size: int,
        n_basis: int = 4,
        r: int = 128,
        n_sinkhorn: int = 5,
        tau: float = 0.05,
        reduce_method: str = None,
    ):
        super().__init__()
        assert L % block_size == 0, (
            f"L={L} must be divisible by block_size={block_size}"
        )
        assert reduce_method in self.REDUCE_METHODS, (
            f"reduce_method must be one of {self.REDUCE_METHODS}, got {reduce_method}"
        )

        self.L = L
        self.block_size = block_size
        self.n_basis = n_basis
        self.r = r
        self.n_sinkhorn = n_sinkhorn
        self.tau = tau
        self.n_blocks = L // block_size
        self.reduce_method = reduce_method

        # ------------------------------------------------------------------
        # Parameter allocation depending on the chosen reduction.
        # ------------------------------------------------------------------
        if reduce_method is None:
            # Full UniMixing-Lite: basis composition for W_B + low-rank for W_G.
            # Z: shared basis matrices, shape (n_basis, block_size, block_size).
            self.Z = nn.Parameter(torch.randn(n_basis, block_size, block_size) * 0.02)
            # omega: per-block combination weights, shape (n_blocks, n_basis).
            self.omega = nn.Parameter(torch.randn(self.n_blocks, n_basis) * 0.02)
            # A_G, B_G: low-rank factors of the global matrix W_G ~= A_G @ B_G.
            self.A_G = nn.Parameter(torch.randn(self.n_blocks, r) * 0.02)
            self.B_G = nn.Parameter(torch.randn(r, self.n_blocks) * 0.02)

        elif reduce_method == "self_attention":
            # Self-Attention reduction:
            #   Local  = X W_V   (single shared W_V across all blocks, no Sinkhorn)
            #   Global = softmax((X W_Q)(X W_K)^T / sqrt(d))  (input-dependent)
            self.W_V = nn.Parameter(torch.randn(block_size, block_size) * 0.02)
            self.W_Q = nn.Parameter(torch.randn(block_size, block_size) * 0.02)
            self.W_K = nn.Parameter(torch.randn(block_size, block_size) * 0.02)

        elif reduce_method == "tokenmixer":
            # TokenMixer reduction:
            #   Local  = X       (identity, no parameters)
            #   Global = G       (fixed permutation matrix, no parameters)
            # The exact block-swap pattern is constructed on the fly in
            # `compute_W_G`. No learnable parameters are allocated here.
            pass

        elif reduce_method == "wukong":
            # Wukong / FM reduction:
            #   Local  = Y       (fixed projection matrix; reduces memory of XX^T)
            #   Global = X I (XI)^T = X X^T  (input-dependent, low-rank via Y)
            self.Y = nn.Parameter(torch.randn(block_size, block_size) * 0.02)

        elif reduce_method == "kda":
            # Kimi Delta Attention (KDA) reduction (arxiv 2510.26692).
            # Under the unified framework (Eq. 4.3.7), KDA is (see summary 6.21):
            #   Local  = X W_V   (per-block V projection, same family as
            #                     self-attention; the "value" that enters
            #                     the recurrent state)
            #   Global = G^KDA(X) (lower-triangular, cumulative path-dependent
            #                      mixing -- see `compute_W_G`)
            #
            # Recurrent state update (Eq.1 of the KDA paper):
            #   S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1}
            #                            + beta_t k_t v_t^T
            #   o_t = S_t^T q_t
            # where k_t = X_t W_K, q_t = X_t W_Q, v_t = X_t W_V (local mixing),
            #       alpha_t = channel-wise forget gate in (0,1)^D (finer-grained
            #                 than GDN's scalar gate; this is KDA's key tweak),
            #       beta_t  = scalar learning rate in (0,1).
            #
            # For simplicity we set d_k = d_v = block_size. Keys are L2-
            # normalized so (I - beta k k^T) is a well-conditioned reflector.
            #
            # NOTE: `compute_W_G` builds G^KDA explicitly via a double loop
            # (naive parallel form) for readability. It does NOT use the
            # DPLR chunkwise optimization (chunk_kda in the paper), which
            # would reduce complexity from O(T^2 D^2) to O(T D^2). The math
            # is identical; only efficiency differs. See summary 6.21.4:
            # DPLR is an implementation detail, NOT a formal extension of
            # Eq.17 -- KDA fits G(X, W_G).V directly.
            self.W_V = nn.Parameter(torch.randn(block_size, block_size) * 0.02)
            self.W_K = nn.Parameter(torch.randn(block_size, block_size) * 0.02)
            self.W_Q = nn.Parameter(torch.randn(block_size, block_size) * 0.02)
            self.W_alpha = nn.Parameter(torch.randn(block_size, block_size) * 0.02)
            self.W_beta = nn.Parameter(torch.randn(block_size, 1) * 0.02)

        # Normalization (RMSNorm) over the flattened embedding dimension L.
        self.norm = RMSNorm(L)

    # ------------------------------------------------------------------
    # Sinkhorn-Knopp
    # ------------------------------------------------------------------
    def sinkhorn(self, M: torch.Tensor) -> torch.Tensor:
        """
        Apply Sinkhorn-Knopp iterations to make M doubly stochastic.

        Implements Eq. 4.3.6 of the paper:
            1. Symmetrize: (M + M^T) / 2
            2. Temperature scaling + exp: ensures all elements are positive
            3. Alternating row / column normalization until both sum to 1

        Args:
            M: tensor of shape (..., n, n). Sinkhorn is applied on the last
               two dimensions, supporting batched inputs.
        Returns:
            Doubly stochastic tensor of the same shape as M.
        """
        # Step 1: symmetrize to satisfy the symmetry constraint.
        M = (M + M.transpose(-1, -2)) / 2
        # Step 2: divide by temperature and exp to make all entries positive.
        M = torch.exp(M / self.tau)
        # Step 3: alternating row / column normalization.
        for _ in range(self.n_sinkhorn):
            M = M / M.sum(dim=-1, keepdim=True)   # row normalization: each row sums to 1
            M = M / M.sum(dim=-2, keepdim=True)   # column normalization: each col sums to 1
        return M

    # ------------------------------------------------------------------
    # Local mixing matrix W_B^{*i}
    # ------------------------------------------------------------------
    def compute_W_B_star(self) -> torch.Tensor:
        """
        Compute the per-block local mixing matrices W_B^{*i}.

        Returns:
            W_B_star: tensor of shape (n_blocks, block_size, block_size).
        """
        if self.reduce_method is None:
            # Basis composition: W_B^{*i} = Sinkhorn(sum_l omega_l^i * Z_l)
            omega_norm = F.softmax(self.omega, dim=-1)  # (n_blocks, n_basis)
            W_B_raw = torch.einsum(
                "nb,bxy->nxy", omega_norm, self.Z
            )  # (n_blocks, block_size, block_size)
            W_B_star = self.sinkhorn(W_B_raw)
            return W_B_star

        elif self.reduce_method == "self_attention":
            # Shared W_V across all blocks (free matrix, no Sinkhorn constraint).
            return (
                self.W_V.unsqueeze(0)
                .expand(self.n_blocks, -1, -1)
                .clone()
            )

        elif self.reduce_method == "tokenmixer":
            # Identity local mixing: each block passes through unchanged.
            identity = torch.eye(self.block_size)
            return identity.unsqueeze(0).expand(self.n_blocks, -1, -1).clone()

        elif self.reduce_method == "wukong":
            # Fixed Y matrix shared across all blocks.
            return (
                self.Y.unsqueeze(0)
                .expand(self.n_blocks, -1, -1)
                .clone()
            )

        elif self.reduce_method == "kda":
            # Shared W_V across all blocks (per-token V projection).
            # Same local-mixing family as self_attention; the V values are
            # what get accumulated into the recurrent state S_t.
            return (
                self.W_V.unsqueeze(0)
                .expand(self.n_blocks, -1, -1)
                .clone()
            )

    # ------------------------------------------------------------------
    # Global mixing matrix W_G
    # ------------------------------------------------------------------
    def compute_W_G(self, X_blocks: torch.Tensor = None) -> torch.Tensor:
        """
        Compute the global mixing matrix W_G (controls inter-block interaction).

        Args:
            X_blocks: tensor of shape (batch, n_blocks, block_size). Required
                      for input-dependent methods (self_attention, wukong).
        Returns:
            W_G: shape (n_blocks, n_blocks) for static methods, or
                 (batch, n_blocks, n_blocks) for input-dependent methods.
        """
        if self.reduce_method is None:
            # Low-rank + Sinkhorn: W_G = Sinkhorn(A_G @ B_G)
            W_G_raw = self.A_G @ self.B_G  # (n_blocks, n_blocks)
            W_G = self.sinkhorn(W_G_raw)
            return W_G

        elif self.reduce_method == "self_attention":
            # softmax((X W_Q)(X W_K)^T / sqrt(d)) -- input-dependent.
            assert X_blocks is not None, "self_attention requires X_blocks"
            Q = torch.einsum("bnx,xy->bny", X_blocks, self.W_Q)
            K = torch.einsum("bnx,xy->bny", X_blocks, self.W_K)
            attn = torch.einsum("bmx,bnx->bmn", Q, K) / (self.block_size ** 0.5)
            W_G = F.softmax(attn, dim=-1)  # (batch, n_blocks, n_blocks)
            return W_G

        elif self.reduce_method == "tokenmixer":
            # Fixed permutation matrix. We use a block-swap pattern
            # (swap adjacent block pairs) that is structurally representative
            # of TokenMixer's transposition behavior (see Appendix A).
            n = self.n_blocks
            perm_indices = torch.arange(n).clone()
            for i in range(0, n, 2):
                if i + 1 < n:
                    tmp = perm_indices[i].clone()
                    perm_indices[i] = perm_indices[i + 1]
                    perm_indices[i + 1] = tmp
            # P[i, j] = 1 if perm_indices[i] == j  =>  rows select source blocks
            W_G = torch.eye(n)[perm_indices]  # (n, n) permutation matrix
            return W_G

        elif self.reduce_method == "wukong":
            # X X^T -- input-dependent, computes pairwise block similarity.
            assert X_blocks is not None, "wukong requires X_blocks"
            XXt = torch.einsum("bmx,bnx->bmn", X_blocks, X_blocks)
            return XXt  # (batch, n_blocks, n_blocks)

        elif self.reduce_method == "kda":
            # G^KDA(X): lower-triangular, cumulative path-dependent global
            # mixing. Under Eq.17 this is the G(X, W_G) factor; the local
            # mixing X W_V (from `compute_W_B_star`) plays the role of V,
            # so forward's `W_G @ H` computes  O = G^KDA . V  exactly.
            # See summary 6.21.1 for the derivation.
            assert X_blocks is not None, "kda requires X_blocks"
            B, T, D = X_blocks.shape  # batch, n_blocks, block_size
            # Per-block projections (d_k = d_v = block_size for simplicity).
            K = torch.einsum("btd,de->bte", X_blocks, self.W_K)  # (B, T, D)
            Q = torch.einsum("btd,de->bte", X_blocks, self.W_Q)  # (B, T, D)
            # L2-normalize keys so (I - beta k k^T) is a stable reflector
            # (keeps the state-transition matrix well-conditioned).
            K = F.normalize(K, dim=-1)
            # Channel-wise forget gate alpha_t in (0,1)^D; scalar beta_t in (0,1).
            alpha = torch.sigmoid(X_blocks @ self.W_alpha)  # (B, T, D)
            beta = torch.sigmoid(X_blocks @ self.W_beta).squeeze(-1)  # (B, T)

            # Naive parallel construction of G^KDA (lower-triangular).
            #   G[t, i] = q_t^T (prod_{j=i+1}^t M_j) k_i   for i <= t, else 0
            #   M_j = (I - beta_j k_j k_j^T) Diag(alpha_j)
            #
            # Accumulate the product right-to-left: starting from i = t with
            # prod = I (empty product), each step does  prod <- prod @ M_i
            # so that for the next (smaller) i the product covers M_{i+1..t}.
            # Verify: i=t -> prod=I, G=q_t k_t; i=t-1 -> prod=M_t, G=q_t M_t k_{t-1};
            #         i=t-2 -> prod=M_t M_{t-1}, G=q_t M_t M_{t-1} k_{t-2}. Correct.
            #
            # NOTE: This is the naive O(T^2 D^2) construction. The paper's
            # DPLR chunkwise form (`chunk_kda`) computes the same G in O(T D^2)
            # but is far less readable. We sacrifice speed for clarity here.
            G = X_blocks.new_zeros(B, T, T)
            eye = torch.eye(D, device=X_blocks.device, dtype=X_blocks.dtype)
            for b in range(B):
                for t in range(T):
                    prod = eye.clone()  # prod_{j=t+1}^t M_j = I (empty)
                    for i in range(t, -1, -1):
                        k_i = K[b, i]            # (D,)
                        a_i = alpha[b, i]        # (D,)
                        beta_i = beta[b, i]      # scalar
                        # G[t, i] = beta_i * q_t^T prod k_i
                        # The beta_i factor comes from the recurrent write
                        # term beta_i k_i v_i^T in Eq.1 (KDA paper Table 1
                        # omits beta for brevity, but the full parallel form
                        # must include it -- else parallel != recurrent).
                        G[b, t, i] = beta_i * (Q[b, t] @ prod @ k_i)
                        # M_i = (I - beta_i k_i k_i^T) Diag(alpha_i)
                        # Factor order matches Eq.1 of the KDA paper:
                        # Diag(alpha) acts first (channel-wise decay of the
                        # carried state), then the Householder-style reflector
                        # (I - beta k k^T) writes/erases along k.
                        M_i = (
                            eye - beta_i * torch.outer(k_i, k_i)
                        ) @ torch.diag(a_i)
                        prod = prod @ M_i  # right-multiply for next (i-1)
            return G  # (B, T, T) lower-triangular

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementing the optimized UniMixing computation pipeline.

        Pipeline (mirrors Eq. 4.3.3 -> 4.3.4 -> 4.3.5 of the paper):
            Step 1 (Eq. 4.3.3): Blocking -- split flatten(X) into L/B blocks
            Step 2 (Eq. 4.3.4): Local mixing -- apply per-block W_B^{*i}
            Step 3 (Eq. 4.3.5): Global mixing -- apply W_G across blocks
            Step 4          : Residual connection + RMSNorm

        Args:
            X: tensor of shape (batch, L). Flattened input embedding.
        Returns:
            output: tensor of shape (batch, L).
        """
        batch = X.shape[0]
        n_blocks, block_size = self.n_blocks, self.block_size

        # Step 1 (Eq. 4.3.3): Blocking -- the explicit Split operation.
        # [x_1 | x_2 | ... | x_{L/B}] = Split(flatten(X), L/B)
        X_blocks = X.view(batch, n_blocks, block_size)  # (batch, n_blocks, block_size)

        # Step 2 (Eq. 4.3.4): Local mixing -- apply W_B^{*i} to each block.
        # H = [x_1 W_B^1 | x_2 W_B^2 | ... | x_{L/B} W_B^{L/B}]
        W_B_star = self.compute_W_B_star()  # (n_blocks, block_size, block_size)
        # Per-block matmul: (batch, n_blocks, B) x (n_blocks, B, B) -> (batch, n_blocks, B)
        H = torch.einsum("bnx,nxy->bny", X_blocks, W_B_star)

        # Step 3 (Eq. 4.3.5): Global mixing -- apply W_G across blocks.
        # out = W_G @ reshape(H, L/B, B)
        W_G = self.compute_W_G(X_blocks)
        if W_G.dim() == 2:
            # Static W_G (UniMixing-Lite, TokenMixer): shared across batch.
            out = torch.einsum("mn,bnB->bmB", W_G, H)
        else:
            # Input-dependent W_G (Self-Attention, Wukong): per-sample.
            out = torch.einsum("bmn,bnB->bmB", W_G, H)

        out = out.reshape(batch, self.L)

        # Step 4: Residual connection + RMSNorm.
        # O = RMSNorm(X + UniMixing-Lite(X))
        output = self.norm(X + out)
        return output

    # ------------------------------------------------------------------
    # Full Kronecker-like mixing matrix W_G ⊗ {W_B^i} for visualization
    # ------------------------------------------------------------------
    def get_mixing_matrices(self, X: torch.Tensor = None):
        """
        Compute and return the full W_G ⊗ {W_B^i} matrix and its components.

        This is the equivalent permutation matrix W^perm from Eq. 4.3.2.
        The (i, j)-th block (each of size block_size x block_size) of the full
        matrix is given by:
            W_full[i*B:(i+1)*B, j*B:(j+1)*B] = W_G[i, j] * W_B^{*j}

        For input-dependent methods (self_attention, wukong), a fixed input X
        is required; if X has a batch dimension, the first sample is used.

        Args:
            X: optional tensor of shape (L,) or (batch, L). Required for
               input-dependent methods (self_attention, wukong).
        Returns:
            W_full: tensor of shape (L, L) -- the full Kronecker-like product.
            W_G   : tensor of shape (n_blocks, n_blocks) -- global mixing matrix.
            W_B   : tensor of shape (n_blocks, block_size, block_size) -- per-block
                    local mixing matrices.
        """
        # Prepare input blocks for input-dependent methods.
        if X is None:
            if self.reduce_method in ("self_attention", "wukong"):
                raise ValueError(
                    f"reduce_method='{self.reduce_method}' requires input X "
                    "to compute the input-dependent W_G."
                )
            X_blocks = None
        else:
            if X.dim() == 1:
                X = X.unsqueeze(0)
            X_blocks = X.view(-1, self.n_blocks, self.block_size)

        # Compute components.
        W_B = self.compute_W_B_star()  # (n_blocks, block_size, block_size)
        W_G = self.compute_W_G(X_blocks)  # (n_blocks, n_blocks) or (batch, n_blocks, n_blocks)

        # Take the first sample for input-dependent W_G (visualization purpose).
        if W_G.dim() == 3:
            W_G = W_G[0]  # (n_blocks, n_blocks)

        # Construct the full L x L matrix.
        # Block (i, j) of W_full is W_G[i, j] * W_B^{*j}.
        W_full = torch.zeros(self.L, self.L)
        for i in range(self.n_blocks):
            for j in range(self.n_blocks):
                W_full[
                    i * self.block_size : (i + 1) * self.block_size,
                    j * self.block_size : (j + 1) * self.block_size,
                ] = W_G[i, j] * W_B[j]

        return W_full, W_G, W_B


# ---------------------------------------------------------------------------
# Visualization: W_G ⊗ W_B for full UniMixing-Lite and three reductions
# ---------------------------------------------------------------------------
def visualize_mixing_matrices(
    L: int = 24,
    block_size: int = 4,
    save_path: str = None,
):
    """
    Visualize W_G ⊗ W_B for full UniMixing-Lite and the three baseline reductions.

    Produces a 5x3 grid:
        - Rows   : UniMixing-Lite / Self-Attention / TokenMixer / Wukong (FM) / KDA
        - Col 1  : W_G (global mixing matrix, n_blocks x n_blocks)
        - Col 2  : stacked W_B^{*i} for the first few blocks (local mixing)
        - Col 3  : W_G ⊗ W_B (the full L x L mixing matrix W^perm)

    Args:
        L          : total embedding dimension.
        block_size : size of each local block.
        save_path  : path to save the figure. If None, defaults to
                     'unimixer_mixing_matrices.png' next to this script.
    """
    import matplotlib.pyplot as plt

    torch.manual_seed(42)
    n_basis = 4
    # Use a small rank for visualization; the paper uses r=128 in practice.
    r = 4

    # A fixed random input for input-dependent methods (self_attention, wukong).
    X = torch.randn(1, L)

    methods = [
        (None, "UniMixing-Lite (full)"),
        ("self_attention", "Self-Attention"),
        ("tokenmixer", "TokenMixer"),
        ("wukong", "Wukong (FM)"),
        ("kda", "KDA (Kimi Delta Attn)"),
    ]

    fig, axes = plt.subplots(5, 3, figsize=(16, 22))

    for row, (method, title) in enumerate(methods):
        model = UniMixingLite(
            L=L,
            block_size=block_size,
            n_basis=n_basis,
            r=r,
            n_sinkhorn=5,
            tau=0.05,
            reduce_method=method,
        )
        model.eval()

        with torch.no_grad():
            W_full, W_G, W_B = model.get_mixing_matrices(X)

        W_full_np = W_full.detach().numpy()
        W_G_np = W_G.detach().numpy()
        W_B_np = W_B.detach().numpy()

        # ---- Column 1: W_G (n_blocks x n_blocks) ----
        ax = axes[row, 0]
        im = ax.imshow(W_G_np, cmap="viridis", aspect="auto")
        ax.set_title(f"{title}\n$W_G$  ({W_G_np.shape[0]}x{W_G_np.shape[1]})", fontsize=11)
        ax.set_xlabel("block j")
        ax.set_ylabel("block i")
        # Annotate cell values for small matrices.
        if W_G_np.shape[0] <= 16:
            for i in range(W_G_np.shape[0]):
                for j in range(W_G_np.shape[1]):
                    ax.text(
                        j, i, f"{W_G_np[i, j]:.2f}",
                        ha="center", va="center",
                        color="white" if W_G_np[i, j] < W_G_np.max() / 2 else "black",
                        fontsize=7,
                    )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # ---- Column 2: stacked W_B^{*i} for the first few blocks ----
        ax = axes[row, 1]
        n_show = min(4, W_B_np.shape[0])
        W_B_show = W_B_np[:n_show].reshape(-1, W_B_np.shape[2])  # (n_show * B, B)
        im = ax.imshow(W_B_show, cmap="viridis", aspect="auto")
        ax.set_title(f"$W_B^{{*i}}$  (first {n_show} blocks stacked)", fontsize=11)
        ax.set_xlabel("feature dim within block")
        ax.set_ylabel("block index (stacked)")
        # Draw horizontal lines to separate blocks.
        for k in range(1, n_show):
            ax.axhline(y=k * W_B_np.shape[2] - 0.5, color="red", linewidth=1.0, linestyle="--")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # ---- Column 3: W_G ⊗ W_B (full L x L) ----
        ax = axes[row, 2]
        im = ax.imshow(W_full_np, cmap="viridis", aspect="auto")
        ax.set_title(
            f"$W_G \\otimes W_B$  ({W_full_np.shape[0]}x{W_full_np.shape[1]})",
            fontsize=11,
        )
        ax.set_xlabel("feature dim")
        ax.set_ylabel("feature dim")
        # Draw grid lines to delineate blocks.
        for k in range(1, model.n_blocks):
            ax.axhline(y=k * block_size - 0.5, color="red", linewidth=0.6, linestyle="--", alpha=0.6)
            ax.axvline(x=k * block_size - 0.5, color="red", linewidth=0.6, linestyle="--", alpha=0.6)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(
        "UniMixing-Lite: $W_G \\otimes W_B$ visualization for different methods\n"
        f"(L={L}, block_size={block_size}, n_basis={n_basis}, r={r}, tau=0.05)",
        fontsize=14,
        y=1.00,
    )
    plt.tight_layout()

    if save_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(script_dir, "unimixer_mixing_matrices.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[viz] Figure saved to: {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# Main: functional tests + visualization
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    
    print("=" * 78)
    print("1. Functional test: full UniMixing-Lite forward pass")
    print("=" * 78)
    torch.manual_seed(0)
    L, block_size = 768, 6
    model = UniMixingLite(
        L=L,
        block_size=block_size,
        n_basis=4,
        r=128,
        n_sinkhorn=5,
        tau=0.05,
        reduce_method=None,
    )
    x = torch.randn(32, L)
    out = model(x)
    print(f"  Input : {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    print()
    print("=" * 78)
    print("2. Reduction tests: UniMixing-Lite degenerates to baselines")
    print("=" * 78)
    # Use a small L for these tests so all methods are comparable.
    L_small, block_size_small = 24, 4
    x_small = torch.randn(8, L_small)
    for method in ["self_attention", "tokenmixer", "wukong", "kda"]:
        m = UniMixingLite(
            L=L_small,
            block_size=block_size_small,
            reduce_method=method,
        )
        out = m(x_small)
        n_params = sum(p.numel() for p in m.parameters())
        print(f"  {method:15s}: input {tuple(x_small.shape)} -> output {tuple(out.shape)}, params={n_params}")

    print()
    print("=" * 78)
    print("3. Visualization: W_G, W_B, and W_G ⊗ W_B for each method")
    print("=" * 78)
    save_path = visualize_mixing_matrices(L=24, block_size=4)
    print(f"  Done. Figure: {save_path}")
