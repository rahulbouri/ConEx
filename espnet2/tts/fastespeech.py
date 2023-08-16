""" FastESpeech """

from typing import Dict
from typing import Sequence
from typing import Tuple

import torch
import torch.nn.functional as F

from typeguard import check_argument_types

from espnet.nets.pytorch_backend.e2e_tts_fastspeech import (
    FeedForwardTransformerLoss as FastSpeechLoss,  # NOQA
)
from espnet.nets.pytorch_backend.fastspeech.duration_predictor import DurationPredictor
from espnet.nets.pytorch_backend.fastspeech.length_regulator import LengthRegulator
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.tacotron2.decoder import Postnet
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.embedding import ScaledPositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder import (
    Encoder as TransformerEncoder,  # noqa: H301
)

from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.torch_utils.initialize import initialize
from espnet2.tts.abs_tts import AbsTTS
from espnet2.tts.prosody_encoder import ProsodyEncoder


class FastESpeech(AbsTTS):
    """FastESpeech module.

    This module adds a VQ-VAE prosody encoder to the FastSpeech model, and
    takes cues from FastSpeech 2 for training.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/abs/1905.09263
    .. _`FastSpeech 2: Fast and High-Quality End-to-End Text to Speech`:
        https://arxiv.org/abs/2006.04558

    Args:
        idim (int): Dimension of the input -> size of the phoneme vocabulary.
        odim (int): Dimension of the output -> dimension of the mel-spectrograms.
        adim (int, optional): Dimension of the phoneme embeddings, dimension of the
        prosody embedding, the hidden size of the self-attention, 1D convolution
        in the FFT block.
        aheads (int, optional): Number of attention heads.
        elayers (int, optional): Number of encoder layers/blocks.
        eunits (int, optional): Number of encoder hidden units
        -> The number of units of position-wise feed forward layer.
        dlayers (int, optional): Number of decoder layers/blocks.
        dunits (int, optional): Number of decoder hidden units
        -> The number of units of position-wise feed forward layer.
        positionwise_layer_type (str, optional):  Type of position-wise feed forward
        layer - linear or conv1d.
        positionwise_conv_kernel_size (int, optional): kernel size of positionwise
        conv1d layer.
        use_scaled_pos_enc (bool, optional):
             Whether to use trainable scaled positional encoding.
        encoder_normalize_before (bool, optional):
            Whether to perform layer normalization before encoder block.
        decoder_normalize_before (bool, optional):
            Whether to perform layer normalization before decoder block.
        encoder_concat_after (bool, optional): Whether to concatenate attention
            layer's input and output in encoder.
        decoder_concat_after (bool, optional): Whether to concatenate attention
            layer's input and output in decoder.
        duration_predictor_layers (int, optional): Number of duration predictor layers.
        duration_predictor_chans (int, optional): Number of duration predictor channels.
        duration_predictor_kernel_size (int, optional):
            Kernel size of duration predictor.
        reduction_factor (int, optional): Factor to multiply with output dimension.
        encoder_type (str, optional): Encoder architecture type.
        decoder_type (str, optional): Decoder architecture type.
        # spk_embed_dim (int, optional): Number of speaker embedding dimensions.
        # spk_embed_integration_type: How to integrate speaker embedding.
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
        # reduction_factor (int, optional): Reduction factor.
        prosody_num_embs (int, optional): The higher this value, the higher the
        capacity in the information bottleneck.
        prosody_hidden_dim (int, optional): Number of hidden channels.
        prosody_emb_integration_type: How to integrate prosody embedding.
        transformer_enc_dropout_rate (float, optional):
            Dropout rate in encoder except attention & positional encoding.
        transformer_enc_positional_dropout_rate (float, optional):
            Dropout rate after encoder positional encoding.
        transformer_enc_attn_dropout_rate (float, optional):
            Dropout rate in encoder self-attention module.
        transformer_dec_dropout_rate (float, optional):
            Dropout rate in decoder except attention & positional encoding.
        transformer_dec_positional_dropout_rate (float, optional):
            Dropout rate after decoder positional encoding.
        transformer_dec_attn_dropout_rate (float, optional):
            Dropout rate in decoder self-attention module.
        duration_predictor_dropout_rate (float, optional):
            Dropout rate in duration predictor.
        init_type (str, optional):
            How to initialize transformer parameters.
        init_enc_alpha (float, optional):
            Initial value of alpha in scaled pos encoding of the encoder.
        init_dec_alpha (float, optional):
            Initial value of alpha in scaled pos encoding of the decoder.
        use_masking (bool, optional):
            Whether to apply masking for padded part in loss calculation.
        use_weighted_masking (bool, optional):
            Whether to apply weighted masking in loss calculation.
    """

    def __init__(
        self,
        # network structure related
        idim: int,
        odim: int,
        adim: int = 384,
        aheads: int = 4,
        elayers: int = 6,
        eunits: int = 1536,
        dlayers: int = 6,
        dunits: int = 1536,
        postnet_layers: int = 0,  # 5
        postnet_chans: int = 512,
        postnet_filts: int = 5,
        positionwise_layer_type: str = "conv1d",
        positionwise_conv_kernel_size: int = 1,
        use_scaled_pos_enc: bool = True,
        use_batch_norm: bool = True,
        encoder_normalize_before: bool = True,
        decoder_normalize_before: bool = True,
        encoder_concat_after: bool = False,
        decoder_concat_after: bool = False,
        duration_predictor_layers: int = 2,
        duration_predictor_chans: int = 384,
        duration_predictor_kernel_size: int = 3,
        reduction_factor: int = 1,
        encoder_type: str = "transformer",
        decoder_type: str = "transformer",
        # # only for conformer
        # conformer_pos_enc_layer_type: str = "rel_pos",
        # conformer_self_attn_layer_type: str = "rel_selfattn",
        # conformer_activation_type: str = "swish",
        # use_macaron_style_in_conformer: bool = True,
        # use_cnn_in_conformer: bool = True,
        # conformer_enc_kernel_size: int = 7,
        # conformer_dec_kernel_size: int = 31,
        # # pretrained spk emb
        # spk_embed_dim: int = None,
        # spk_embed_integration_type: str = "add",
        # reference encoder
        ref_enc_conv_layers: int = 2,
        ref_enc_conv_chans_list: Sequence[int] = (32, 32),
        ref_enc_conv_kernel_size: int = 3,
        ref_enc_conv_stride: int = 1,
        ref_enc_gru_layers: int = 1,
        ref_enc_gru_units: int = 32,
        ref_emb_integration_type: str = "add",
        # prosody encoder
        prosody_num_embs: int = 256,
        prosody_hidden_dim: int = 128,
        prosody_emb_integration_type: str = "add",
        # training related
        transformer_enc_dropout_rate: float = 0.1,
        transformer_enc_positional_dropout_rate: float = 0.1,
        transformer_enc_attn_dropout_rate: float = 0.1,
        transformer_dec_dropout_rate: float = 0.1,
        transformer_dec_positional_dropout_rate: float = 0.1,
        transformer_dec_attn_dropout_rate: float = 0.1,
        duration_predictor_dropout_rate: float = 0.1,
        postnet_dropout_rate: float = 0.5,
        init_type: str = "xavier_uniform",
        init_enc_alpha: float = 1.0,
        init_dec_alpha: float = 1.0,
        use_masking: bool = False,
        use_weighted_masking: bool = False,
    ):
        """Initialize FastESpeech module."""
        assert check_argument_types()
        super().__init__()

        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.eos = idim - 1
        self.reduction_factor = reduction_factor
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.use_scaled_pos_enc = use_scaled_pos_enc
        self.prosody_emb_integration_type = prosody_emb_integration_type
        # self.spk_embed_dim = spk_embed_dim
        # if self.spk_embed_dim is not None:
        #     self.spk_embed_integration_type = spk_embed_integration_type

        # use idx 0 as padding idx, see:
        # https://stackoverflow.com/questions/61172400/what-does-padding-idx-do-in-nn-embeddings
        self.padding_idx = 0

        # get positional encoding class
        pos_enc_class = (
            ScaledPositionalEncoding if self.use_scaled_pos_enc else PositionalEncoding
        )

        # define encoder
        encoder_input_layer = torch.nn.Embedding(
            num_embeddings=idim, embedding_dim=adim, padding_idx=self.padding_idx
        )
        if encoder_type == "transformer":
            self.encoder = TransformerEncoder(
                idim=idim,
                attention_dim=adim,
                attention_heads=aheads,
                linear_units=eunits,
                num_blocks=elayers,
                input_layer=encoder_input_layer,
                dropout_rate=transformer_enc_dropout_rate,
                positional_dropout_rate=transformer_enc_positional_dropout_rate,
                attention_dropout_rate=transformer_enc_attn_dropout_rate,
                pos_enc_class=pos_enc_class,
                normalize_before=encoder_normalize_before,
                concat_after=encoder_concat_after,
                positionwise_layer_type=positionwise_layer_type,
                positionwise_conv_kernel_size=positionwise_conv_kernel_size,
            )
        # elif encoder_type == "conformer":
        #     self.encoder = ConformerEncoder(
        #         idim=idim,
        #         attention_dim=adim,
        #         attention_heads=aheads,
        #         linear_units=eunits,
        #         num_blocks=elayers,
        #         input_layer=encoder_input_layer,
        #         dropout_rate=transformer_enc_dropout_rate,
        #         positional_dropout_rate=transformer_enc_positional_dropout_rate,
        #         attention_dropout_rate=transformer_enc_attn_dropout_rate,
        #         normalize_before=encoder_normalize_before,
        #         concat_after=encoder_concat_after,
        #         positionwise_layer_type=positionwise_layer_type,
        #         positionwise_conv_kernel_size=positionwise_conv_kernel_size,
        #         macaron_style=use_macaron_style_in_conformer,
        #         pos_enc_layer_type=conformer_pos_enc_layer_type,
        #         selfattention_layer_type=conformer_self_attn_layer_type,
        #         activation_type=conformer_activation_type,
        #         use_cnn_module=use_cnn_in_conformer,
        #         cnn_module_kernel=conformer_enc_kernel_size,
        #     )
        else:
            raise ValueError(f"{encoder_type} is not supported.")

        # define additional projection for prosody embedding
        if self.prosody_emb_integration_type == "concat":
            self.prosody_projection = torch.nn.Linear(
                adim * 2, adim
            )

        # define prosody encoder
        self.prosody_encoder = ProsodyEncoder(
            odim,
            adim=adim,
            num_embeddings=prosody_num_embs,
            hidden_dim=prosody_hidden_dim,
            ref_enc_conv_layers=ref_enc_conv_layers,
            ref_enc_conv_chans_list=ref_enc_conv_chans_list,
            ref_enc_conv_kernel_size=ref_enc_conv_kernel_size,
            ref_enc_conv_stride=ref_enc_conv_stride,
            global_enc_gru_layers=ref_enc_gru_layers,
            global_enc_gru_units=ref_enc_gru_units,
            global_emb_integration_type=ref_emb_integration_type,
        )

        # # define additional projection for speaker embedding
        # if self.spk_embed_dim is not None:
        #     if self.spk_embed_integration_type == "add":
        #         self.projection = torch.nn.Linear(self.spk_embed_dim, adim)
        #     else:
        #         self.projection = torch.nn.Linear(adim + self.spk_embed_dim, adim)

        # define duration predictor
        self.duration_predictor = DurationPredictor(
            idim=adim,
            n_layers=duration_predictor_layers,
            n_chans=duration_predictor_chans,
            kernel_size=duration_predictor_kernel_size,
            dropout_rate=duration_predictor_dropout_rate,
        )

        # define length regulator
        self.length_regulator = LengthRegulator()

        # define decoder
        # NOTE: we use encoder as decoder
        # because fastspeech's decoder is the same as encoder
        if decoder_type == "transformer":
            self.decoder = TransformerEncoder(
                idim=0,
                attention_dim=adim,
                attention_heads=aheads,
                linear_units=dunits,
                num_blocks=dlayers,
                input_layer=None,
                dropout_rate=transformer_dec_dropout_rate,
                positional_dropout_rate=transformer_dec_positional_dropout_rate,
                attention_dropout_rate=transformer_dec_attn_dropout_rate,
                pos_enc_class=pos_enc_class,
                normalize_before=decoder_normalize_before,
                concat_after=decoder_concat_after,
                positionwise_layer_type=positionwise_layer_type,
                positionwise_conv_kernel_size=positionwise_conv_kernel_size,
            )
        # elif decoder_type == "conformer":
        #     self.decoder = ConformerEncoder(
        #         idim=0,
        #         attention_dim=adim,
        #         attention_heads=aheads,
        #         linear_units=dunits,
        #         num_blocks=dlayers,
        #         input_layer=None,
        #         dropout_rate=transformer_dec_dropout_rate,
        #         positional_dropout_rate=transformer_dec_positional_dropout_rate,
        #         attention_dropout_rate=transformer_dec_attn_dropout_rate,
        #         normalize_before=decoder_normalize_before,
        #         concat_after=decoder_concat_after,
        #         positionwise_layer_type=positionwise_layer_type,
        #         positionwise_conv_kernel_size=positionwise_conv_kernel_size,
        #         macaron_style=use_macaron_style_in_conformer,
        #         pos_enc_layer_type=conformer_pos_enc_layer_type,
        #         selfattention_layer_type=conformer_self_attn_layer_type,
        #         activation_type=conformer_activation_type,
        #         use_cnn_module=use_cnn_in_conformer,
        #         cnn_module_kernel=conformer_dec_kernel_size,
        #     )
        else:
            raise ValueError(f"{decoder_type} is not supported.")

        # define final projection
        self.feat_out = torch.nn.Linear(adim, odim * reduction_factor)

        # define postnet
        self.postnet = (
            None
            if postnet_layers == 0
            else Postnet(
                idim=idim,
                odim=odim,
                n_layers=postnet_layers,
                n_chans=postnet_chans,
                n_filts=postnet_filts,
                use_batch_norm=use_batch_norm,
                dropout_rate=postnet_dropout_rate,
            )
        )

        # initialize parameters
        self._reset_parameters(
            init_type=init_type,
            init_enc_alpha=init_enc_alpha,
            init_dec_alpha=init_dec_alpha,
        )

        # define criterions
        self.criterion = FastSpeechLoss(
            use_masking=use_masking, use_weighted_masking=use_weighted_masking
        )

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        durations: torch.Tensor,
        durations_lengths: torch.Tensor,
        spembs: torch.Tensor = None,
        train_ar_prior: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Calculate forward propagation.

        Args:
            text (LongTensor): Batch of padded token ids (B, Tmax).
            text_lengths (LongTensor): Batch of lengths of each input (B,).
            speech (Tensor): Batch of padded target features (B, Lmax, odim).
            speech_lengths (LongTensor): Batch of the lengths of each target (B,).
            durations (LongTensor): Batch of padded durations (B, Tmax + 1).
            durations_lengths (LongTensor): Batch of duration lengths (B, Tmax + 1).
            spembs (Tensor, optional): Batch of speaker embeddings (B, spk_embed_dim).

        Returns:
            Tensor: Loss scalar value.
            Dict: Statistics to be monitored.
            Tensor: Weight value.

        """
        # train_ar_prior = True  # TC marker
        text = text[:, : text_lengths.max()]  # for data-parallel
        speech = speech[:, : speech_lengths.max()]  # for data-parallel
        durations = durations[:, : durations_lengths.max()]  # for data-parallel

        batch_size = text.size(0)

        # Add eos at the last of sequence
        xs = F.pad(text, [0, 1], "constant", self.padding_idx)
        for i, l in enumerate(text_lengths):
            xs[i, l] = self.eos
        ilens = text_lengths + 1

        ys, ds = speech, durations
        olens = speech_lengths

        # forward propagation
        before_outs, after_outs, d_outs, ref_embs, \
            vq_loss, ar_prior_loss, perplexity = self._forward(
                xs,
                ilens,
                ys,
                olens,
                ds,
                spembs=spembs,
                is_inference=False,
                train_ar_prior=train_ar_prior
            )

        # modify mod part of groundtruth
        if self.reduction_factor > 1:
            olens = olens.new([olen - olen % self.reduction_factor for olen in olens])
            max_olen = max(olens)
            ys = ys[:, :max_olen]

        if self.postnet is None:
            after_outs = None

        # calculate loss  TODO: refactor if freezing works
        l1_loss, duration_loss = self.criterion(
            after_outs, before_outs, d_outs, ys, ds, ilens, olens
        )
        if train_ar_prior:
            loss = ar_prior_loss
            stats = dict(
                l1_loss=l1_loss.item(),
                duration_loss=duration_loss.item(),
                vq_loss=vq_loss.item(),
                ar_prior_loss=ar_prior_loss.item(),
                loss=loss.item(),
                perplexity=perplexity.item(),
            )
        else :
            loss = l1_loss + duration_loss + vq_loss
            stats = dict(
                l1_loss=l1_loss.item(),
                duration_loss=duration_loss.item(),
                vq_loss=vq_loss.item(),
                loss=loss.item(),
                perplexity=perplexity.item()
            )

        # report extra information
        if self.encoder_type == "transformer" and self.use_scaled_pos_enc:
            stats.update(
                encoder_alpha=self.encoder.embed[-1].alpha.data.item(),
            )
        if self.decoder_type == "transformer" and self.use_scaled_pos_enc:
            stats.update(
                decoder_alpha=self.decoder.embed[-1].alpha.data.item(),
            )

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def _forward(
        self,
        xs: torch.Tensor,
        ilens: torch.Tensor,
        ys: torch.Tensor = None,
        olens: torch.Tensor = None,
        ds: torch.Tensor = None,
        spembs: torch.Tensor = None,
        ref_embs: torch.Tensor = None,
        is_inference: bool = False,
        train_ar_prior: bool = False,
        ar_prior_inference: bool = False,
        alpha: float = 1.0,
        fg_inds: torch.Tensor = None,
    ) -> Sequence[torch.Tensor]:
        # forward encoder
        x_masks = self._source_mask(ilens)
        hs, _ = self.encoder(xs, x_masks)  # (B, Tmax, adim)

        # # integrate speaker embedding
        # if self.spk_embed_dim is not None:
        #     hs = self._integrate_with_spk_embed(hs, spembs)

        # integrate with prosody encoder
        # (B, Tmax, adim)
        p_embs, vq_loss, ar_prior_loss, perplexity, ref_embs = self.prosody_encoder(
            ys,
            ds,
            hs,
            global_embs=ref_embs,
            train_ar_prior=train_ar_prior,
            ar_prior_inference=ar_prior_inference,
            fg_inds=fg_inds,
        )

        hs = self._integrate_with_prosody_embs(hs, p_embs)

        # forward duration predictor
        d_masks = make_pad_mask(ilens).to(xs.device)

        if is_inference:
            d_outs = self.duration_predictor.inference(hs, d_masks)  # (B, Tmax)
            hs = self.length_regulator(hs, d_outs, alpha)  # (B, Lmax, adim)
        else:
            d_outs = self.duration_predictor(hs, d_masks)
            # use groundtruth in training
            hs = self.length_regulator(hs, ds)  # (B, Lmax, adim)

        # forward decoder
        if olens is not None and not is_inference:
            if self.reduction_factor > 1:
                olens_in = olens.new([olen // self.reduction_factor for olen in olens])
            else:
                olens_in = olens
            h_masks = self._source_mask(olens_in)
        else:
            h_masks = None
        zs, _ = self.decoder(hs, h_masks)  # (B, Lmax, adim)
        before_outs = self.feat_out(zs).view(
            zs.size(0), -1, self.odim
        )  # (B, Lmax, odim)

        # postnet -> (B, Lmax//r * r, odim)
        if self.postnet is None:
            after_outs = before_outs
        else:
            after_outs = before_outs + self.postnet(
                before_outs.transpose(1, 2)
            ).transpose(1, 2)

        return before_outs, after_outs, d_outs, ref_embs, vq_loss, ar_prior_loss, \
            perplexity

    def inference(
        self,
        text: torch.Tensor,
        speech: torch.Tensor = None,
        spembs: torch.Tensor = None,
        durations: torch.Tensor = None,
        ref_embs: torch.Tensor = None,
        alpha: float = 1.0,
        use_teacher_forcing: bool = False,
        ar_prior_inference: bool = False,
        fg_inds: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate the sequence of features given the sequences of characters.

        Args:
            text (LongTensor): Input sequence of characters (T,).
            speech (Tensor, optional): Feature sequence to extract style (B, idim).
            spembs (Tensor, optional): Speaker embedding vector (spk_embed_dim,).
            durations (LongTensor, optional): Groundtruth of duration (T + 1,).
            ref_embs (Tensor, optional): Reference embedding vector (B, gru_units).
            alpha (float, optional): Alpha to control the speed.
            use_teacher_forcing (bool, optional): Whether to use teacher forcing.
                If true, groundtruth of duration will be used.

        Returns:
            Tensor: Output sequence of features (L, odim).
            None: Dummy for compatibility.
            None: Dummy for compatibility.

        """
        x, y = text, speech
        spemb, d = spembs, durations

        # add eos at the last of sequence
        x = F.pad(x, [0, 1], "constant", self.eos)

        # setup batch axis
        ilens = torch.tensor([x.shape[0]], dtype=torch.long, device=x.device)
        xs, ys = x.unsqueeze(0), None
        if y is not None:
            ys = y.unsqueeze(0)
        if spemb is not None:
            spembs = spemb.unsqueeze(0)
        if ref_embs is not None:
            ref_embs = ref_embs.unsqueeze(0)

        if use_teacher_forcing:
            # use groundtruth of duration
            ds = d.unsqueeze(0)
            _, after_outs, _, ref_embs, _, ar_prior_loss, _ = self._forward(
                xs,
                ilens,
                ys,
                ds=ds,
                spembs=spembs,
                ref_embs=ref_embs,
                ar_prior_inference=ar_prior_inference,
            )  # (1, L, odim)
        else:
            _, after_outs, _, ref_embs, _, ar_prior_loss, _ = self._forward(
                xs,
                ilens,
                ys,
                spembs=spembs,
                ref_embs=ref_embs,
                is_inference=True,
                alpha=alpha,
                ar_prior_inference=ar_prior_inference,
                fg_inds=fg_inds,
            )  # (1, L, odim)

        return after_outs[0], None, None, ref_embs, ar_prior_loss

    # def _integrate_with_spk_embed(
    #     self, hs: torch.Tensor, spembs: torch.Tensor
    # ) -> torch.Tensor:
    #     """Integrate speaker embedding with hidden states.

    #     Args:
    #         hs (Tensor): Batch of hidden state sequences (B, Tmax, adim).
    #         spembs (Tensor): Batch of speaker embeddings (B, spk_embed_dim).

    #     Returns:
    #         Tensor: Batch of integrated hidden state sequences (B, Tmax, adim).

    #     """
    #     if self.spk_embed_integration_type == "add":
    #         # apply projection and then add to hidden states
    #         spembs = self.projection(F.normalize(spembs))
    #         hs = hs + spembs.unsqueeze(1)
    #     elif self.spk_embed_integration_type == "concat":
    #         # concat hidden states with spk embeds and then apply projection
    #         spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
    #         hs = self.projection(torch.cat([hs, spembs], dim=-1))
    #     else:
    #         raise NotImplementedError("support only add or concat.")

    #     return hs

    def _source_mask(self, ilens: torch.Tensor) -> torch.Tensor:
        """Make masks for self-attention.

        Args:
            ilens (LongTensor): Batch of lengths (B,).

        Returns:
            Tensor: Mask tensor for self-attention.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)

        Examples:
            >>> ilens = [5, 3]
            >>> self._source_mask(ilens)
            tensor([[[1, 1, 1, 1, 1],
                     [1, 1, 1, 0, 0]]], dtype=torch.uint8)

        """
        x_masks = make_non_pad_mask(ilens).to(next(self.parameters()).device)
        return x_masks.unsqueeze(-2)

    def _integrate_with_prosody_embs(
        self, hs: torch.Tensor, p_embs: torch.Tensor
    ) -> torch.Tensor:
        """Integrate prosody embeddings with hidden states.

        Args:
            hs (Tensor): Batch of hidden state sequences (B, Tmax, adim).
            p_embs (Tensor): Batch of prosody embeddings (B, Tmax, adim).

        Returns:
            Tensor: Batch of integrated hidden state sequences (B, Tmax, adim).

        """
        if self.prosody_emb_integration_type == "add":
            # apply projection and then add to hidden states
            # (B, Tmax, adim)
            hs = hs + p_embs
        elif self.prosody_emb_integration_type == "concat":
            # concat hidden states with prosody embeds and then apply projection
            # (B, Tmax, adim)
            hs = self.prosody_projection(torch.cat([hs, p_embs], dim=-1))
        else:
            raise NotImplementedError("support only add or concat.")

        return hs

    def _reset_parameters(
        self, init_type: str, init_enc_alpha: float, init_dec_alpha: float
    ):
        # initialize parameters
        if init_type != "pytorch":
            initialize(self, init_type)

        # initialize alpha in scaled positional encoding
        if self.encoder_type == "transformer" and self.use_scaled_pos_enc:
            self.encoder.embed[-1].alpha.data = torch.tensor(init_enc_alpha)
        if self.decoder_type == "transformer" and self.use_scaled_pos_enc:
            self.decoder.embed[-1].alpha.data = torch.tensor(init_dec_alpha)
