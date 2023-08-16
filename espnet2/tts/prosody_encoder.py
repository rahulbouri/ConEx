from typing import Sequence

import math

import torch
from torch import nn
from torch.nn import functional as F

from typeguard import check_argument_types


class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self,
                 num_embeddings: int,
                 hidden_dim: int,
                 beta: float = 0.25):
        super().__init__()
        self.K = num_embeddings
        self.D = hidden_dim
        self.beta = 0.05  # beta override

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.normal_(0.8, 0.1)  # override

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        # latents = latents.permute(0, 2, 1).contiguous()  # (B, D, L) -> (B, L, D)
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # (BL, D)

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - \
            2 * torch.matmul(flat_latents, self.embedding.weight.t())  # (BL, K)

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1)  # (BL)
        output_inds = encoding_inds.view(latents_shape[0], latents_shape[1])  # (B, L)
        encoding_inds = encoding_inds.unsqueeze(1)  # (BL, 1)

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # (BL, K)

        # Quantize the latents
        # (BL, D)
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)
        quantized_latents = quantized_latents.view(latents_shape)  # (B, L, D)

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        # print(output_inds)
        # print(quantized_latents)

        # The perplexity a useful value to track during training.
        # It indicates how many codes are 'active' on average.
        avg_probs = torch.mean(encoding_one_hot, dim=0)
        # Exponential entropy
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized_latents, vq_loss, output_inds, self.embedding, perplexity


class ProsodyEncoder(nn.Module):
    """VQ-VAE prosody encoder module.

    Args:
        odim (int): Number of input channels (mel spectrogram channels).
        ref_enc_conv_layers (int, optional):
            The number of conv layers in the reference encoder.
        ref_enc_conv_chans_list: (Sequence[int], optional):
            List of the number of channels of conv layers in the referece encoder.
        ref_enc_conv_kernel_size (int, optional):
            Kernal size of conv layers in the reference encoder.
        ref_enc_conv_stride (int, optional):
            Stride size of conv layers in the reference encoder.
        ref_enc_gru_layers (int, optional):
            The number of GRU layers in the reference encoder.
        ref_enc_gru_units (int, optional):
            The number of GRU units in the reference encoder.
        ref_emb_integration_type: How to integrate reference embedding.
        adim (int, optional): This value is not that important.
        This will not change the capacity in the information-bottleneck.
        num_embeddings (int, optional): The higher this value, the higher the
        capacity in the information bottleneck.
        hidden_dim (int, optional): Number of hidden channels.
    """
    def __init__(
        self,
        odim: int,
        adim: int = 64,
        num_embeddings: int = 10,
        hidden_dim: int = 3,
        beta: float = 0.25,
        ref_enc_conv_layers: int = 2,
        ref_enc_conv_chans_list: Sequence[int] = (32, 32),
        ref_enc_conv_kernel_size: int = 3,
        ref_enc_conv_stride: int = 1,
        global_enc_gru_layers: int = 1,
        global_enc_gru_units: int = 32,
        global_emb_integration_type: str = "add",
    ) -> None:
        assert check_argument_types()
        super().__init__()

        # store hyperparameters
        self.global_emb_integration_type = global_emb_integration_type

        padding = (ref_enc_conv_kernel_size - 1) // 2

        self.ref_encoder = RefEncoder(
            ref_enc_conv_layers=ref_enc_conv_layers,
            ref_enc_conv_chans_list=ref_enc_conv_chans_list,
            ref_enc_conv_kernel_size=ref_enc_conv_kernel_size,
            ref_enc_conv_stride=ref_enc_conv_stride,
            ref_enc_conv_padding=padding,
        )

        # get the number of ref enc output units
        ref_enc_output_units = odim
        for i in range(ref_enc_conv_layers):
            ref_enc_output_units = (
                ref_enc_output_units - ref_enc_conv_kernel_size + 2 * padding
            ) // ref_enc_conv_stride + 1
        ref_enc_output_units *= ref_enc_conv_chans_list[-1]

        self.fg_encoder = FGEncoder(
            ref_enc_output_units + global_enc_gru_units,
            hidden_dim=hidden_dim,
        )

        self.global_encoder = GlobalEncoder(
            ref_enc_output_units,
            global_enc_gru_layers=global_enc_gru_layers,
            global_enc_gru_units=global_enc_gru_units,
        )

        # define a projection for the global embeddings
        if self.global_emb_integration_type == "add":
            self.global_projection = nn.Linear(global_enc_gru_units, adim)
        else:
            self.global_projection = nn.Linear(
                adim + global_enc_gru_units, adim
            )

        self.ar_prior = ARPrior(
            adim,
            num_embeddings=num_embeddings,
            hidden_dim=hidden_dim,
        )

        self.vq_layer = VectorQuantizer(num_embeddings, hidden_dim, beta)

        # define a projection for the quantized fine-grained embeddings
        self.qfg_projection = nn.Linear(hidden_dim, adim)

    def forward(
        self,
        ys: torch.Tensor,
        ds: torch.Tensor,
        hs: torch.Tensor,
        global_embs: torch.Tensor = None,
        train_ar_prior: bool = False,
        ar_prior_inference: bool = False,
        fg_inds: torch.Tensor = None,
    ) -> Sequence[torch.Tensor]:
        """Calculate forward propagation.

        Args:
            ys (Tensor): Batch of padded target features (B, Lmax, odim).
            ds (LongTensor): Batch of padded durations (B, Tmax).
            hs (Tensor): Batch of phoneme embeddings (B, Tmax, D).
            global_embs (Tensor, optional): Global embeddings (B, D)

        Returns:
            Tensor: Fine-grained quantized prosody embeddings (B, Tmax, adim).
            Tensor: VQ loss.
            Tensor: Global prosody embeddings (B, ref_enc_gru_units)
        """
        if ys is not None:
            print('generating global_embs')
            ref_embs = self.ref_encoder(ys)  # (B, L', ref_enc_output_units)
            global_embs = self.global_encoder(ref_embs)  # (B, ref_enc_gru_units)

        if ar_prior_inference:
            print('Using ar prior')
            hs_integrated = self._integrate_with_global_embs(hs, global_embs)
            qs, top_inds = self.ar_prior.inference(
                hs_integrated, fg_inds, self.vq_layer.embedding
            )

            qs = self.qfg_projection(qs)  # (B, Tmax, adim)
            assert hs.size(2) == qs.size(2)

            p_embs = self._integrate_with_global_embs(qs, global_embs)
            assert hs.shape == p_embs.shape

            return p_embs, 0, 0, 0, top_inds  # (B, Tmax, adim)

        # concat global embs to ref embs
        global_embs_expanded = global_embs.unsqueeze(1).expand(-1, ref_embs.size(1), -1)
        # (B, Tmax, D)
        ref_embs_integrated = torch.cat([ref_embs, global_embs_expanded], dim=-1)

        # (B, Tmax, hidden_dim)
        fg_embs = self.fg_encoder(ref_embs_integrated, ds, ys.size(1))

        # (B, Tmax, hidden_dim)
        qs, vq_loss, inds, codebook, perplexity = self.vq_layer(fg_embs)
        # Vector quantization should maintain length
        assert hs.size(1) == qs.size(1)

        qs = self.qfg_projection(qs)  # (B, Tmax, adim)
        assert hs.size(2) == qs.size(2)

        p_embs = self._integrate_with_global_embs(qs, global_embs)
        assert hs.shape == p_embs.shape

        ar_prior_loss = 0
        if train_ar_prior:
            # (B, Tmax, adim)
            hs_integrated = self._integrate_with_global_embs(hs, global_embs)
            qs, ar_prior_loss = self.ar_prior(hs_integrated, inds, codebook)
            qs = self.qfg_projection(qs)  # (B, Tmax, adim)
            assert hs.size(2) == qs.size(2)

            p_embs = self._integrate_with_global_embs(qs, global_embs)
            assert hs.shape == p_embs.shape

        return p_embs, vq_loss, ar_prior_loss, perplexity, global_embs

    def _integrate_with_global_embs(
        self,
        qs: torch.Tensor,
        global_embs: torch.Tensor
    ) -> torch.Tensor:
        """Integrate ref embedding with spectrogram hidden states.

        Args:
            qs (Tensor): Batch of quantized FG embeddings (B, Tmax, adim).
            global_embs (Tensor): Batch of global embeddings (B, global_enc_gru_units).

        Returns:
            Tensor: Batch of integrated hidden state sequences (B, Tmax, adim).
        """
        if self.global_emb_integration_type == "add":
            # apply projection to hidden states
            global_embs = self.global_projection(global_embs)
            res = qs + global_embs.unsqueeze(1)
        elif self.global_emb_integration_type == "concat":
            # concat hidden states with prosody embeds and then apply projection
            # (B, Tmax, ref_emb_dim)
            global_embs = global_embs.unsqueeze(1).expand(-1, qs.size(1), -1)
            # (B, Tmax, D)
            res = self.prosody_projection(torch.cat([qs, global_embs], dim=-1))
        else:
            raise NotImplementedError("support only add or concat.")

        return res


class RefEncoder(nn.Module):
    def __init__(
        self,
        ref_enc_conv_layers: int = 2,
        ref_enc_conv_chans_list: Sequence[int] = (32, 32),
        ref_enc_conv_kernel_size: int = 3,
        ref_enc_conv_stride: int = 1,
        ref_enc_conv_padding: int = 1,
    ):
        """Initilize reference encoder module."""
        assert check_argument_types()
        super().__init__()

        # check hyperparameters are valid
        assert ref_enc_conv_kernel_size % 2 == 1, "kernel size must be odd."
        assert (
            len(ref_enc_conv_chans_list) == ref_enc_conv_layers
        ), "the number of conv layers and length of channels list must be the same."

        convs = []
        for i in range(ref_enc_conv_layers):
            conv_in_chans = 1 if i == 0 else ref_enc_conv_chans_list[i - 1]
            conv_out_chans = ref_enc_conv_chans_list[i]
            convs += [
                nn.Conv2d(
                    conv_in_chans,
                    conv_out_chans,
                    kernel_size=ref_enc_conv_kernel_size,
                    stride=ref_enc_conv_stride,
                    padding=ref_enc_conv_padding,
                ),
                nn.ReLU(inplace=True),

            ]
        self.convs = nn.Sequential(*convs)

    def forward(self, ys: torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            ys (Tensor): Batch of padded target features (B, Lmax, odim).

        Returns:
            Tensor: Batch of spectrogram hiddens (B, L', ref_enc_output_units)

        """
        B = ys.size(0)
        ys = ys.unsqueeze(1)  # (B, 1, Lmax, odim)
        hs = self.convs(ys)  # (B, conv_out_chans, L', odim')
        hs = hs.transpose(1, 2)  # (B, L', conv_out_chans, odim')
        L = hs.size(1)
        # (B, L', ref_enc_output_units) -> "flatten"
        hs = hs.contiguous().view(B, L, -1)

        return hs


class GlobalEncoder(nn.Module):
    """Module that creates a global embedding from a hidden spectrogram sequence.

    Args:
    """
    def __init__(
        self,
        ref_enc_output_units: int,
        global_enc_gru_layers: int = 1,
        global_enc_gru_units: int = 32,
    ):
        super().__init__()
        self.gru = torch.nn.GRU(ref_enc_output_units, global_enc_gru_units,
                                global_enc_gru_layers, batch_first=True)

    def forward(
        self,
        hs: torch.Tensor,
    ):
        """Calculate forward propagation.

        Args:
            hs (Tensor): Batch of spectrogram hiddens (B, L', ref_enc_output_units).

        Returns:
            Tensor: Reference embedding (B, ref_enc_gru_units).
        """
        self.gru.flatten_parameters()
        _, global_embs = self.gru(hs)  # (gru_layers, B, ref_enc_gru_units)
        global_embs = global_embs[-1]  # (B, ref_enc_gru_units)

        return global_embs


class FGEncoder(nn.Module):
    """Spectrogram to phoneme alignment module.

    Args:
    """
    def __init__(
        self,
        input_units: int,
        hidden_dim: int = 3,
    ):
        assert check_argument_types()
        super().__init__()

        self.projection = nn.Sequential(
            nn.Sequential(
                nn.Linear(input_units, input_units // 2),
                nn.ReLU(),
                nn.Dropout(p=0.2),
            ),
            nn.Sequential(
                nn.Linear(input_units // 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=0.2),
            )
        )

    def forward(
        self,
        hs: torch.Tensor,
        ds: torch.Tensor,
        Lmax: int
    ):
        """Calculate forward propagation.

        Args:
            hs (Tensor): Batch of spectrogram hiddens
            (B, L', ref_enc_output_units + global_enc_gru_units).
            ds (LongTensor): Batch of padded durations (B, Tmax).

        Returns:
            Tensor: aligned spectrogram hiddens (B, Tmax, hidden_dim).
        """
        # (B, Tmax, ref_enc_output_units + global_enc_gru_units)
        hs = self._align_durations(hs, ds, Lmax)
        hs = self.projection(hs)  # (B, Tmax, hidden_dim)

        return hs

    def _align_durations(self, hs, ds, Lmax):
        """Transform the spectrogram hiddens according to the ground-truth durations
        so that there's only one hidden per phoneme hidden.

        Args:
            # (B, L', ref_enc_output_units + global_enc_gru_units)
            hs (Tensor): Batch of spectrogram hidden state sequences .
            ds (LongTensor): Batch of padded durations (B, Tmax)

        Returns:
            # (B, Tmax, ref_enc_output_units + global_enc_gru_units)
            Tensor: Batch of averaged spectrogram hidden state sequences.
        """
        B = hs.size(0)
        L = hs.size(1)
        D = hs.size(2)

        Tmax = ds.size(1)  # -1 if Tmax + 1

        device = hs.device
        hs_res = torch.zeros(
            [B, Tmax, D],
            device=device
        )  # (B, Tmax, D)

        with torch.no_grad():
            for b_i in range(B):
                durations = ds[b_i]
                multiplier = L / Lmax
                i = 0
                for d_i in range(Tmax):
                    # take into account downsampling because of conv layers
                    d = max(math.floor(durations[d_i].item() * multiplier), 1)
                    if durations[d_i].item() > 0:
                        hs_slice = hs[b_i, i:i + d, :]  # (d, D)
                        hs_res[b_i, d_i, :] = torch.mean(hs_slice, 0)
                        i += d
        hs_res.requires_grad_(hs.requires_grad)
        return hs_res


class ARPrior(nn.Module):
    #  torch.topk(decoder_output, beam_width)
    """Autoregressive prior.

    This module is inspired by the AR prior described in `Generating diverse and
    natural text-to-speech samples using a quantized fine-grained VAE and
    auto-regressive prosody prior`. This prior is fit in the continuous latent space.
    """
    def __init__(
        self,
        adim: int,
        num_embeddings: int = 10,
        hidden_dim: int = 3,
    ):
        assert check_argument_types()
        super().__init__()

        # store hyperparameters
        self.adim = adim
        self.hidden_dim = hidden_dim
        self.num_embeddings = num_embeddings

        self.qs_projection = nn.Linear(hidden_dim, adim)

        self.lstm = nn.LSTMCell(
            self.adim,
            self.num_embeddings,
        )

        self.criterion = nn.NLLLoss()

    def inds_to_embs(self, inds, codebook, device):
        """Returns the quantized embeddings from the codebook,
        corresponding to the indices.

        Args:
            inds (Tensor): Batch of indices (B, Tmax, 1).
            codebook (Embedding): (num_embeddings, D).

        Returns:
            Tensor: Quantized embeddings (B, Tmax, D).
        """
        flat_inds = torch.flatten(inds).unsqueeze(1)  # (BL, 1)

        # Convert to one-hot encodings
        encoding_one_hot = torch.zeros(
            flat_inds.size(0),
            self.num_embeddings,
            device=device
        )
        encoding_one_hot.scatter_(1, flat_inds, 1)  # (BL, K)

        # Quantize the latents
        # (BL, D)
        quantized_embs = torch.matmul(encoding_one_hot, codebook.weight)
        # (B, L, D)
        quantized_embs = quantized_embs.view(
            inds.size(0), inds.size(1), self.hidden_dim
        )

        return quantized_embs

    def top_embeddings(self, emb_scores: torch.Tensor, codebook):
        """Returns the top quantized embeddings from the codebook using the scores.

        Args:
            emb_scores (Tensor): Batch of embedding scores (B, Tmax, num_embeddings).
            codebook (Embedding): (num_embeddings, D).

        Returns:
            Tensor: Top quantized embeddings (B, Tmax, D).
            Tensor: Top 3 inds (B, Tmax, 3).
        """
        _, top_inds = emb_scores.topk(1, dim=-1)  # (B, L, 1)
        quantized_embs = self.inds_to_embs(
            top_inds,
            codebook,
            emb_scores.device,
        )
        _, top3_inds = emb_scores.topk(3, dim=-1)  # (B, L, 1)
        return quantized_embs, top3_inds

    def _forward(self, hs_ref_embs, codebook, fg_inds=None):
        inds = []
        scores = []
        embs = []

        if fg_inds is not None:
            init_embs = self.inds_to_embs(fg_inds, codebook, hs_ref_embs.device)
            embs = [init_emb.unsqueeze(1) for init_emb in init_embs.transpose(1, 0)]

        start = fg_inds.size(1) if fg_inds is not None else 0
        hidden = hs_ref_embs.new_zeros(hs_ref_embs.size(0), self.lstm.hidden_size)
        cell = hs_ref_embs.new_zeros(hs_ref_embs.size(0), self.lstm.hidden_size)

        for i in range(start, hs_ref_embs.size(1)):
            # (B, adim)
            input = hs_ref_embs[:, i]
            if i != 0:
                # (B, 1, adim)
                qs = self.qs_projection(embs[-1])
                # (B, adim)
                input = hs_ref_embs[:, i] + qs.squeeze()
            hidden, cell = self.lstm(input, (hidden, cell))  # (B, K)
            out = hidden.unsqueeze(1)  # (B, 1, K)
            # (B, 1, K)
            emb_scores = F.log_softmax(out, dim=2)
            quantized_embs, top_inds = self.top_embeddings(emb_scores, codebook)
            # (B, 1, hidden_dim)
            embs.append(quantized_embs)
            scores.append(emb_scores)
            inds.append(top_inds)

        out_embs = torch.cat(embs, dim=1)  # (B, L, hidden_dim)
        assert(out_embs.size(0) == hs_ref_embs.size(0))
        assert(out_embs.size(1) == hs_ref_embs.size(1))
        out_emb_scores = torch.cat(scores, dim=1) if start < hs_ref_embs.size(1) else scores
        out_inds = torch.cat(inds, dim=1) if start < hs_ref_embs.size(1) else fg_inds

        return out_embs, out_emb_scores, out_inds

    def forward(self, hs_ref_embs, inds, codebook):
        """Calculate forward propagation.

        Args:
            hs_p_embs (Tensor): Batch of phoneme embeddings
            with integrated global prosody embeddings (B, Tmax, D).
            inds (Tensor): Batch of ground-truth codebook indices
                (B, Tmax).

        Returns:
            Tensor: Batch of predicted quantized latents (B, Tmax, D).
            Tensor: Cross entropy loss value.

        """
        quantized_embs, emb_scores, _ = self._forward(hs_ref_embs, codebook)
        emb_scores = emb_scores.permute(0, 2, 1).contiguous()  # (B, num_embeddings, L)
        loss = self.criterion(emb_scores, inds)
        return quantized_embs, loss

    def inference(self, hs_ref_embs, fg_inds, codebook):
        """Inference duration.

        Args:
            hs_p_embs (Tensor): Batch of phoneme embeddings
            with integrated global prosody embeddings (B, Tmax, D).

        Returns:
            Tensor: Batch of predicted quantized latents (B, Tmax, D).

        """
        # Random sampling
        # fg_inds = torch.rand(hs_ref_embs.size(0), hs_ref_embs.size(1))
        # fg_inds *= codebook.weight.size(0) - 1
        # fg_inds = torch.round(fg_inds)
        # fg_inds = fg_inds.long()

        quantized_embs, _, top_inds = self._forward(hs_ref_embs, codebook, fg_inds)
        return quantized_embs, top_inds
