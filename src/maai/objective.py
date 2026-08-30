import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange

from typing import Dict, List, Tuple, Union


def bin_times_to_frames(bin_times: List[float], frame_hz: float) -> List[int]:
    """Convert a list of time durations into a list of frame counts.
    
    Args:
        bin_times (List[float]): A list of time durations in seconds.
        frame_hz (float): The frame rate (Hz) of the system.
        
    Returns:
        List[int]: A list of corresponding frame counts.
    """
    frames = torch.tensor(bin_times, dtype=torch.float32) * float(frame_hz)
    frames = torch.floor(frames + 0.5).clamp(min=1)
    return frames.to(dtype=torch.long).tolist()


def bin_times_to_frames_cumulative(bin_times: List[float], frame_hz: float) -> List[int]:
    """Convert bin durations into frame counts by rounding the bin *boundaries*.

    Unlike :func:`bin_times_to_frames`, which rounds every duration on its own,
    this rounds the cumulative boundary of each bin and takes the difference, so
    the bins tile the horizon exactly (``sum(bin_frames) / frame_hz`` equals
    ``sum(bin_times)``).

    The single-channel VAP model (``vap_mono``) is trained with this rule and its
    ``p_now`` / ``p_future`` are weighted by the bin lengths, so the two rules are
    not interchangeable there. With ``bin_times=[0.2, 0.4, 0.6, 0.8]`` they agree
    at 10/20/50 Hz and differ at 12.5 Hz ([3, 5, 7, 10] here versus [3, 5, 8, 10]).

    Args:
        bin_times (List[float]): A list of bin durations in seconds.
        frame_hz (float): The frame rate (Hz) of the system.

    Returns:
        List[int]: A list of corresponding frame counts.
    """
    if frame_hz <= 0:
        raise ValueError(f"frame_hz must be > 0, got {frame_hz}")

    bin_frames: List[int] = []
    cumulative_time = 0.0
    previous_boundary = 0
    for bin_time in bin_times:
        cumulative_time += float(bin_time)
        boundary = int(math.floor(cumulative_time * float(frame_hz) + 0.5))
        frames = boundary - previous_boundary
        if frames <= 0:
            raise ValueError(
                f"bin_times {bin_times} with frame_hz={frame_hz} produce a non-positive bin width"
            )
        bin_frames.append(frames)
        previous_boundary = boundary
    return bin_frames


class ProjectionWindow:
    """Extracts and evaluates projection windows to determine voice activity.
    
    Used to chunk future voice activity sequences into discrete projection bins.
    """
    def __init__(
        self,
        bin_times: List = [0.2, 0.4, 0.6, 0.8],
        frame_hz: float = 50,
        threshold_ratio: float = 0.5,
        num_channels: int = 2,
    ):
        """Initialize the ProjectionWindow.
        
        Args:
            bin_times (List[float]): Duration of each projection bin in seconds.
            frame_hz (float): Frame rate of the input signal.
            threshold_ratio (float): Ratio threshold to consider a bin as active.
            num_channels (int): Number of speaker channels the projection covers
                (2 for the standard VAP model, 1 for the single-channel one).
        """
        super().__init__()
        self.bin_times = bin_times
        self.frame_hz = frame_hz
        self.threshold_ratio = threshold_ratio
        self.num_channels = num_channels

        self.bin_frames = (
            bin_times_to_frames_cumulative(bin_times, frame_hz)
            if num_channels == 1
            else bin_times_to_frames(bin_times, frame_hz)
        )
        self.n_bins = len(self.bin_frames)
        self.total_bins = self.n_bins * num_channels
        self.horizon = sum(self.bin_frames)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(\n"
        s += f"  bin_times: {self.bin_times}\n"
        s += f"  bin_frames: {self.bin_frames}\n"
        s += f"  frame_hz: {self.frame_hz}\n"
        s += f"  num_channels: {self.num_channels}\n"
        s += f"  thresh: {self.threshold_ratio}\n"
        s += ")\n"
        return s

    def projection(self, va: Tensor) -> Tensor:
        """
        Extract projection (bins)
        (b, n, c) -> (b, N, c, M), M=horizon window size, N=valid frames

        Arguments:
            va:         Tensor (B, N, C)

        Returns:
            vaps:       Tensor (B, m, C, M)

        """
        # Shift to get next frame projections
        return va[..., 1:, :].unfold(dimension=-2, size=sum(self.bin_frames), step=1)

    def projection_bins(self, projection_window: Tensor) -> Tensor:
        """
        Iterate over the bin boundaries and sum the activity
        for each channel/speaker.
        divide by the number of frames to get activity ratio.
        If ratio is greater than or equal to the threshold_ratio
        the bin is considered active
        """

        start = 0
        v_bins = []
        for b in self.bin_frames:
            end = start + b
            m = projection_window[..., start:end].sum(dim=-1) / b
            m = (m >= self.threshold_ratio).float()
            v_bins.append(m)
            start = end
        return torch.stack(v_bins, dim=-1)  # (*, t, c, n_bins)

    def __call__(self, va: Tensor) -> Tensor:
        projection_windows = self.projection(va)
        return self.projection_bins(projection_windows)


class Codebook(nn.Module):
    """A discrete codebook that maps binary sequences to indices and vice-versa.
    
    Represents combinations of future voice activity patterns.
    """
    def __init__(self, bin_frames, num_channels: int = 2):
        """Initialize the Codebook.
        
        Args:
            bin_frames (List[int]): List of frame counts for each bin.
            num_channels (int): Number of speaker channels encoded per state
                (2 for the standard VAP model, 1 for the single-channel one).
        """
        super().__init__()
        self.bin_frames = bin_frames
        self.num_channels: int = num_channels
        self.n_bins: int = len(self.bin_frames)
        self.total_bins: int = self.n_bins * num_channels
        self.n_classes: int = 2 ** self.total_bins

        self.emb = nn.Embedding(
            num_embeddings=self.n_classes, embedding_dim=self.total_bins
        )
        self.emb.weight.data = self.create_code_vectors(self.total_bins)
        self.emb.weight.requires_grad_(False)

    def single_idx_to_onehot(self, idx: int, d: int = 8) -> Tensor:
        assert idx < 2 ** d, "must be possible with {d} binary digits"
        z = torch.zeros(d)
        b = bin(idx).replace("0b", "")
        for i, v in enumerate(b[::-1]):
            z[i] = float(v)
        return z

    def create_code_vectors(self, n_bins: int) -> Tensor:
        """
        Create a matrix of all one-hot encodings representing a binary sequence of `self.total_bins` places
        Useful for usage in `nn.Embedding` like module.
        """
        n_codes = 2 ** n_bins
        embs = torch.zeros((n_codes, n_bins))
        for i in range(2 ** n_bins):
            embs[i] = self.single_idx_to_onehot(i, d=n_bins)
        return embs

    def encode(self, x: Tensor) -> Tensor:
        """

        Encodes projection_windows x (*, 2, 4) to indices in codebook (..., 1)

        Arguments:
            x:          Tensor (*, 2, 4)

        Inspiration for distance calculation:
            https://github.com/lucidrains/vector-quantize-pytorch/blob/master/vector_quantize_pytorch/vector_quantize_pytorch.py
        """
        assert x.shape[-2:] == (
            self.num_channels,
            self.n_bins,
        ), f"Codebook expects (..., {self.num_channels}, {self.n_bins}) got {x.shape}"

        # compare with codebook and get closest idx
        shape = x.shape
        flatten = rearrange(
            x, "... c bpp -> (...) (c bpp)", c=self.num_channels, bpp=self.n_bins
        )
        embed = self.emb.weight.T
        dist = -(
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )
        embed_ind = dist.max(dim=-1).indices
        embed_ind = embed_ind.view(*shape[:-2])
        return embed_ind

    def decode(self, idx: Tensor):
        v = self.emb(idx)
        return rearrange(v, "... (c b) -> ... c b", c=self.num_channels)

    def forward(self, projection_windows: Tensor):
        return self.encode(projection_windows)


class ObjectiveVAP(nn.Module):
    """The central objective module for Voice Activity Projection (VAP).
    
    Handles the transformation of raw future voice activities into codebook labels,
    calculates probabilities, and computes the loss for predictions.
    """
    def __init__(
        self,
        bin_times: List[float] = [0.2, 0.4, 0.6, 0.8],
        frame_hz: float = 50,
        threshold_ratio: float = 0.5,
        num_channels: int = 2,
    ):
        """Initialize the ObjectiveVAP module.
        
        Args:
            bin_times (List[float]): Bin durations in seconds.
            frame_hz (float): Frame rate.
            threshold_ratio (float): Threshold to mark a bin as active.
            num_channels (int): Number of speaker channels covered by one
                codebook state. 2 (the default) gives the standard two-speaker
                objective with ``2 ** (2 * n_bins)`` classes; 1 gives the
                single-channel objective used by ``vap_mono`` with
                ``2 ** n_bins`` classes.
        """
        super().__init__()
        self.frame_hz = frame_hz
        self.bin_times = bin_times
        self.num_channels = num_channels
        # The single-channel model is trained with boundary-rounded bins and
        # weights p_now / p_future by the bin lengths, so the two rules are not
        # interchangeable there (they differ at 12.5 Hz).
        self.bin_frames: List[int] = (
            bin_times_to_frames_cumulative(bin_times, frame_hz)
            if num_channels == 1
            else bin_times_to_frames(bin_times, frame_hz)
        )
        self.horizon = sum(self.bin_frames)
        self.horizon_time = sum(bin_times)

        self.codebook = Codebook(self.bin_frames, num_channels=num_channels)
        self.projection_window_extractor = ProjectionWindow(
            bin_times, frame_hz, threshold_ratio, num_channels=num_channels
        )
        self.requires_grad_(False)

        self.lid_n_classes = 3

    def __repr__(self):
        s = str(self.__class__.__name__)
        s += f"\n{self.codebook}"
        s += f"\n{self.projection_window_extractor}"
        s += "\n"
        return s

    @property
    def n_classes(self) -> int:
        return self.codebook.n_classes

    @property
    def n_bins(self) -> int:
        return self.codebook.n_bins

    def probs_self_activity(
        self,
        probs: Tensor,
        from_bin: int = 0,
        to_bin: int = 3,
        scale_with_bins: bool = True,
    ) -> Tensor:
        """Single-channel P(this speaker is active) over bins ``[from_bin, to_bin]``.

        Only meaningful for a ``num_channels=1`` objective. There is no other
        speaker to normalize against, so the returned value is the expectation
        of the (frame-weighted) activity ratio and is already a probability --
        unlike :meth:`probs_next_speaker_aggregate`, nothing is renormalized.

        Args:
            probs (Tensor): (B, n_frames, n_classes) softmax over the codebook.
            from_bin (int): First bin of the range (inclusive).
            to_bin (int): Last bin of the range (inclusive).
            scale_with_bins (bool): Weight each bin by its length in frames.

        Returns:
            Tensor: (B, n_frames) expected activity ratio in [0, 1].
        """
        assert (
            probs.ndim == 3
        ), f"Expected probs of shape (B, n_frames, n_classes) but got {probs.shape}"

        idx = torch.arange(self.codebook.n_classes, device=probs.device)
        states = self.codebook.decode(idx)  # (n_classes, 1, n_bins)
        a = states[:, 0, from_bin : to_bin + 1]  # (n_classes, k)
        if scale_with_bins:
            w = torch.tensor(
                self.bin_frames[from_bin : to_bin + 1],
                dtype=a.dtype,
                device=a.device,
            )
        else:
            w = torch.ones(a.shape[-1], dtype=a.dtype, device=a.device)
        ratio = (a * w).sum(-1) / w.sum()
        return torch.einsum("bid,d->bi", probs, ratio)

    def probs_bins(self, probs: Tensor) -> Tensor:
        """Single-channel per-bin marginal activity probability.

        Args:
            probs (Tensor): (B, n_frames, n_classes) softmax over the codebook.

        Returns:
            Tensor: (B, n_frames, n_bins) marginal probability per bin.
        """
        assert (
            probs.ndim == 3
        ), f"Expected probs of shape (B, n_frames, n_classes) but got {probs.shape}"

        idx = torch.arange(self.codebook.n_classes, device=probs.device)
        states = self.codebook.decode(idx)[:, 0, :]  # (n_classes, n_bins)
        return torch.einsum("bid,dk->bik", probs, states)

    def probs_next_speaker_aggregate(
        self,
        probs: Tensor,
        from_bin: int = 0,
        to_bin: int = 3,
        scale_with_bins: bool = False,
    ) -> Tensor:
        assert (
            probs.ndim == 3
        ), f"Expected probs of shape (B, n_frames, n_classes) but got {probs.shape}"
        idx = torch.arange(self.codebook.n_classes).to(probs.device)
        states = self.codebook.decode(idx)

        if scale_with_bins:
            states = states * torch.tensor(self.bin_frames)
        abp = states[:, :, from_bin : to_bin + 1].sum(-1)  # sum speaker activity bins
        # Dot product over all states
        p_all = torch.einsum("bid,dc->bic", probs, abp)
        # normalize
        p_all /= p_all.sum(-1, keepdim=True) + 1e-5
        return p_all

    def probs_speaker_bin_aggregate(
        self,
        probs: Tensor,
        from_bin: int = 0,
        to_bin: int = 3,
        scale_with_bins: bool = False,
    ) -> Tensor:
        """
        Aggregate discrete VAP state probabilities into per-speaker, per-bin
        expected activity.

        Args:
            probs: (B, n_frames, n_classes) probabilities over codebook states.
            from_bin/to_bin: inclusive bin range to return.
            scale_with_bins: if True, scales each bin by its length in frames
                (i.e., returns expected active frames rather than probability).

        Returns:
            Tensor of shape (B, n_frames, 2, n_bins_selected) where the last
            dimension corresponds to bins [from_bin..to_bin].
        """
        assert (
            probs.ndim == 3
        ), f"Expected probs of shape (B, n_frames, n_classes) but got {probs.shape}"
        if from_bin < 0 or to_bin >= self.n_bins or from_bin > to_bin:
            raise ValueError(
                f"Invalid bin range: from_bin={from_bin}, to_bin={to_bin}, n_bins={self.n_bins}"
            )

        idx = torch.arange(self.codebook.n_classes, device=probs.device)
        states = self.codebook.decode(idx)  # (n_classes, 2, n_bins)

        states = states[:, :, from_bin : to_bin + 1]  # (n_classes, 2, n_bins_selected)
        if scale_with_bins:
            bin_frames = torch.tensor(
                self.bin_frames[from_bin : to_bin + 1], device=probs.device, dtype=states.dtype
            )
            states = states * bin_frames  # broadcast over (n_classes, 2, n_bins_selected)

        # Expected activity per speaker/bin under the state distribution
        # (B, n_frames, n_classes) x (n_classes, 2, n_bins_selected) -> (B, n_frames, 2, n_bins_selected)
        p_bins = torch.einsum("bid,dcn->bicn", probs, states)
        return p_bins

    def window_to_win_dialog_states(self, wins):
        return (wins.sum(-1) > 0).sum(-1)

    def get_labels(self, va: Tensor) -> Tensor:
        projection_windows = self.projection_window_extractor(va).type(va.dtype)
        idx = self.codebook(projection_windows)
        return idx
    
    def get_labels_bc(self, bc_frame: Tensor) -> Tensor:
        
        #
        # bc_frame: (B, N_FRAMES)
        #
        
        #print(bc_frame.shape)
        PROJECTION_SHIFT_SIZE = 0.5 # Seconds
        shift_size = int(PROJECTION_SHIFT_SIZE * self.frame_hz)
        APPEND_SIZE = int(2.0 * self.frame_hz)
        bc_projection_frame = torch.zeros(
            bc_frame.shape[0],
            bc_frame.shape[1] - APPEND_SIZE,
            dtype=bc_frame.dtype,
            device=bc_frame.device
        )  
        for b in range(bc_frame.shape[0]):
            for i in range(shift_size, bc_frame.shape[1] - APPEND_SIZE):
                bc_projection_frame[b, i-shift_size] = bc_frame[b, i]
        
        return bc_projection_frame

    def get_da_labels(self, va: Tensor) -> Tuple[Tensor, Tensor]:
        projection_windows = self.projection_window_extractor(va).type(va.dtype)
        idx = self.codebook(projection_windows)
        ds = self.window_to_win_dialog_states(projection_windows)
        return idx, ds

    def loss_vap(
        self, logits: Tensor, labels: Tensor, reduction: str = "mean"
    ) -> Tensor:
        assert (
            logits.ndim == 3
        ), f"Exptected logits of shape (B, N_FRAMES, N_CLASSES) but got {logits.shape}"
        assert (
            labels.ndim == 2
        ), f"Exptected labels of shape (B, N_FRAMES) but got {labels.shape}"

        nmax = labels.shape[1]
        if logits.shape[1] > nmax:
            logits = logits[:, :nmax]

        # CrossEntropyLoss over discrete labels
        loss = F.cross_entropy(
            rearrange(logits, "b n d -> (b n) d"),
            rearrange(labels, "b n -> (b n)"),
            reduction=reduction,
        )
        # Shape back to original shape if reduction != 'none'
        if reduction == "none":
            loss = rearrange(loss, "(b n) -> b n", n=nmax)
        return loss
    
    def loss_lid(
        self, logits: Tensor, labels: Tensor, reduction: str = "mean"
    ) -> Tensor:
        assert (
            logits.ndim == 3
        ), f"Exptected logits of shape (B, N_FRAMES, N_CLASSES) but got {logits.shape}"
        assert (
            labels.ndim == 2
        ), f"Exptected labels of shape (B, N_FRAMES) but got {labels.shape}"

        nmax = labels.shape[1]
        if logits.shape[1] > nmax:
            logits = logits[:, :nmax]

        # CrossEntropyLoss over discrete labels
        loss = F.cross_entropy(
            rearrange(logits, "b n d -> (b n) d"),
            rearrange(labels, "b n -> (b n)"),
            reduction=reduction,
        )
        # Shape back to original shape if reduction != 'none'
        if reduction == "none":
            loss = rearrange(loss, "(b n) -> b n", n=nmax)
        
        return loss

    def loss_bc(self, bc_output, bc_label, bc_positive_weight=1.0):
        return F.binary_cross_entropy_with_logits(bc_output, bc_label, pos_weight=torch.tensor([bc_positive_weight], device=bc_output.device))
    
    def loss_vad(self, vad_output, vad):
        n = vad_output.shape[-2]
        return F.binary_cross_entropy_with_logits(vad_output, vad[:, :n])
    
    def loss_vad_mono(self, vad_output, vad):
        n = vad_output.shape[-2]
        v = vad[:, :n, 1]
        # print(torch.squeeze(vad_output))
        # print(v.shape)
        # n = 1
        return F.binary_cross_entropy_with_logits(torch.squeeze(vad_output), v)

    def get_probs(self, logits: Tensor) -> Dict[str, Tensor]:
        """
        Extracts labels from the voice-activity, va.
        The labels are based on projections of the future and so the valid
        frames with corresponding labels are strictly less then the original number of frams.

        Arguments:
        -----------
        logits:     torch.Tensor (B, N_FRAMES, N_CLASSES)
        va:         torch.Tensor (B, N_FRAMES, 2)

        Return:
        -----------
            Dict[probs, p, p_bc, labels]  which are all torch.Tensors
        """

        assert (
            logits.shape[-1] == self.n_classes
        ), f"Logits have wrong shape. {logits.shape} != (..., {self.n_classes}) that is (B, N_FRAMES, N_CLASSES)"

        probs = logits.softmax(dim=-1)

        return {
            "probs": probs,
            "p_now": self.probs_next_speaker_aggregate(
                probs=probs, from_bin=0, to_bin=1
            ),
            "p_future": self.probs_next_speaker_aggregate(
                probs=probs, from_bin=2, to_bin=3
            ),
            "p_tot": self.probs_next_speaker_aggregate(
                probs=probs, from_bin=0, to_bin=3
            ),
        }

    @torch.no_grad()
    def extract_prediction_and_targets(
        self,
        p_now: Tensor,
        p_fut: Tensor,
        events: Dict[str, List[List[Tuple[int, int, int]]]],
        device=None,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        batch_size = len(events["hold"])

        preds = {"hs": [], "hs2": [], "pred_shift": [], "pred_shift2": [], "ls": [], "pred_backchannel": [], "pred_backchannel2": [], "lid": []}
        targets = {"hs": [], "hs2": [], "pred_shift": [], "pred_shift2": [], "ls": [], "pred_backchannel": [], "pred_backchannel2": [], "lid": []}

        for b in range(batch_size):
            ###########################################
            # Hold vs Shift
            ###########################################
            # The metrics (i.e. shift/hold) are binary so we must decide
            # which 'class' corresponds to which numeric label
            # we use Holds=0, Shifts=1
            for start, end, speaker in events["shift"][b]:
                pshift = p_now[b, start:end, speaker]
                preds["hs"].append(pshift)
                targets["hs"].append(torch.ones_like(pshift))

            for start, end, speaker in events["hold"][b]:
                phold = 1 - p_now[b, start:end, speaker]
                preds["hs"].append(phold)
                targets["hs"].append(torch.zeros_like(phold))

            ###########################################
            # Hold vs Shift ver2
            ###########################################
            # The metrics (i.e. shift/hold) are binary so we must decide
            # which 'class' corresponds to which numeric label
            # we use Holds=0, Shifts=1
            for start, end, speaker in events["shift"][b]:
                pshift = p_now[b, start:end, speaker]
                # preds["hs"].append(pshift)
                # targets["hs"].append(torch.ones_like(pshift))
                preds["hs2"].append(torch.tensor([torch.mean(pshift)]))
                targets["hs2"].append(torch.ones(1))

            for start, end, speaker in events["hold"][b]:
                phold = 1 - p_now[b, start:end, speaker]
                # preds["hs"].append(phold)
                # targets["hs"].append(torch.zeros_like(phold))

                preds["hs2"].append(torch.tensor([torch.mean(phold)]))
                targets["hs2"].append(torch.zeros(1))

            ###########################################
            # Shift-prediction
            ###########################################
            for start, end, speaker in events["pred_shift"][b]:
                # prob of next speaker -> the correct next speaker i.e. a SHIFT
                pshift = p_fut[b, start:end, speaker]
                preds["pred_shift"].append(pshift)
                targets["pred_shift"].append(torch.ones_like(pshift))
            for start, end, speaker in events["pred_shift_neg"][b]:
                # prob of next speaker -> the correct next speaker i.e. a HOLD
                phold = 1 - p_fut[b, start:end, speaker]  # 1-shift = Hold
                preds["pred_shift"].append(phold)
                # Negatives are zero -> hold predictions
                targets["pred_shift"].append(torch.zeros_like(phold))
            
            ###########################################
            # Shift-prediction ver2
            ###########################################
            for start, end, speaker in events["pred_shift"][b]:
                # prob of next speaker -> the correct next speaker i.e. a SHIFT
                pshift = p_fut[b, start:end, speaker]
                preds["pred_shift2"].append(torch.tensor([torch.mean(pshift)]))
                targets["pred_shift2"].append(torch.ones(1))
            for start, end, speaker in events["pred_shift_neg"][b]:
                # prob of next speaker -> the correct next speaker i.e. a HOLD
                phold = 1 - p_fut[b, start:end, speaker]  # 1-shift = Hold
                preds["pred_shift2"].append(torch.tensor([torch.mean(phold)]))
                targets["pred_shift2"].append(torch.zeros(1))
            
            ###########################################
            # Backchannel-prediction
            ###########################################
            # TODO: Backchannel with p_now/p_fut???
            p_bc = p_now
            for start, end, speaker in events["pred_backchannel"][b]:
                # prob of next speaker -> the correct next backchanneler i.e. a Backchannel
                pred_bc = p_bc[b, start:end, speaker]
                preds["pred_backchannel"].append(pred_bc)
                targets["pred_backchannel"].append(torch.ones_like(pred_bc))
            for start, end, speaker in events["pred_backchannel_neg"][b]:
                # prob of 'speaker' making a 'backchannel' in the close future
                # over these negatives this probability should be low -> 0
                # so no change of probability have to be made (only the labels are now zero)
                pred_bc = p_bc[b, start:end, speaker]  # 1-shift = Hold
                preds["pred_backchannel"].append(
                    pred_bc
                )  # Negatives are zero -> hold predictions
                targets["pred_backchannel"].append(torch.zeros_like(pred_bc))
            
            ###########################################
            # Backchannel-prediction ver2
            ###########################################
            # TODO: Backchannel with p_now/p_fut???
            p_bc = p_now
            for start, end, speaker in events["pred_backchannel"][b]:
                # prob of next speaker -> the correct next backchanneler i.e. a Backchannel
                pred_bc = p_bc[b, start:end, speaker]
                preds["pred_backchannel2"].append(torch.tensor([torch.mean(pred_bc)]))
                targets["pred_backchannel2"].append(torch.ones(1))
            for start, end, speaker in events["pred_backchannel_neg"][b]:
                # prob of 'speaker' making a 'backchannel' in the close future
                # over these negatives this probability should be low -> 0
                # so no change of probability have to be made (only the labels are now zero)
                pred_bc = p_bc[b, start:end, speaker]  # 1-shift = Hold
                preds["pred_backchannel2"].append(torch.tensor([torch.mean(pred_bc)]))
                targets["pred_backchannel2"].append(torch.zeros(1))
            
            ###########################################
            # Long vs Short
            ###########################################
            # TODO: Should this be the same as backchannel
            # or simply next speaker probs?
            for start, end, speaker in events["long"][b]:
                # prob of next speaker -> the correct next speaker i.e. a LONG
                plong = p_fut[b, start:end, speaker]
                preds["ls"].append(plong)
                targets["ls"].append(torch.ones_like(plong))
            for start, end, speaker in events["short"][b]:
                # the speaker in the 'short' events is the speaker who
                # utters a short utterance: p[b, start:end, speaker] means:
                # the  speaker saying something short has this probability
                # of continue as a 'long'
                # Therefore to correctly predict a 'short' entry this probability
                # should be low -> 0
                # thus we do not have to subtract the prob from 1 (only the labels are now zero)
                # prob of next speaker -> the correct next speaker i.e. a SHORT
                pshort = p_fut[b, start:end, speaker]  # 1-shift = Hold
                preds["ls"].append(pshort)
                # Negatives are zero -> short predictions
                targets["ls"].append(torch.zeros_like(pshort))

        # cat/stack/flatten to single tensor
        device = device if device is not None else p_now.device
        out_preds = {}
        out_targets = {}
        for k, v in preds.items():
            if len(v) > 0:
                out_preds[k] = torch.cat(v).to(device)
            else:
                out_preds[k] = None
        for k, v in targets.items():
            if len(v) > 0:
                out_targets[k] = torch.cat(v).long().to(device)
            else:
                out_targets[k] = None
        return out_preds, out_targets

    @torch.no_grad()
    def extract_prediction_and_targets_bc(
        self,
        p_bc: Tensor,
        events: Dict[str, List[List[Tuple[int, int, int]]]],
        device=None,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        
        batch_size = p_bc.shape[0]

        preds = {"pred_bc": []}
        targets = {"pred_bc": []}
        
        #print(events)

        for b in range(batch_size):
            
            if len(events["pred_bc"][b]) == 0:
                continue
            
            for start, end, label in events["pred_bc"][b]:
                p_ = p_bc[b, start:end]
                preds["pred_bc"].append(torch.tensor([torch.mean(p_)]))
                
                targets["pred_bc"].append(torch.ones(1))
        
        for b in range(batch_size):
            
            if len(events["pred_bc_negative"][b]) == 0:
                continue
            
            for start, end, label in events["pred_bc_negative"][b]:
                p_ = p_bc[b, start:end]
                preds["pred_bc"].append(torch.tensor([torch.mean(p_)]))
                
                targets["pred_bc"].append(torch.zeros(1))

        # # cat/stack/flatten to single tensor
        # device = device if device is not None else p_bc.device
        # out_preds = {}
        # out_targets = {}
        # for k, v in preds.items():
        #     if len(v) > 0:
        #         out_preds[k] = torch.cat(v).to(device)
        #     else:
        #         out_preds[k] = None
        # for k, v in targets.items():
        #     if len(v) > 0:
        #         out_targets[k] = torch.cat(v).long().to(device)
        #     else:
        #         out_targets[k] = None
        
        out_preds = preds
        out_targets = targets
        
        return out_preds, out_targets

    @torch.no_grad()
    def match_bc_events(
        self,
        events_prediction: Tensor,
        events_gt: Tensor,
        threshold_sec: float = 0.3,
    ):
        
        threshold_delay_frame = int(self.frame_hz * threshold_sec)

        preds = {"pred_bc": []}
        targets = {"pred_bc": []}

        
        for b in range(len(events_gt)):
            
            dict_done_pred = {}
            for start, end, _ in events_gt[b]:
                
                hit_pred = False
                for start_pred, end_pred, _ in events_prediction[b]:
                    
                    hit = False    
                    if start - threshold_delay_frame <= start_pred <= end + threshold_delay_frame:
                        hit = True
                    if start - threshold_delay_frame <= end_pred <= end + threshold_delay_frame:
                        hit = True
                    if start_pred <= start and end_pred >= end:
                        hit = True
                    
                    if hit:
                        hit_pred = True
                        dict_done_pred[(start_pred, end_pred)] = True
                    
                if hit_pred:
                    preds["pred_bc"].append(torch.ones(1))
                    targets["pred_bc"].append(torch.ones(1))
                else:
                    preds["pred_bc"].append(torch.zeros(1))
                    targets["pred_bc"].append(torch.ones(1))
            
            #print("dict_done_pred", dict_done_pred)
            
            for start_pred, end_pred, _ in events_prediction[b]:
                
                if (start_pred, end_pred) not in dict_done_pred:
                    preds["pred_bc"].append(torch.ones(1))
                    targets["pred_bc"].append(torch.zeros(1))

        out_preds = preds
        out_targets = targets
        
        return out_preds, out_targets

if __name__ == "__main__":
    ob = ObjectiveVAP()
    print(ob)
    events_prediction = [
        [(10,15,0), (20,25,0), (30,35,0)],
    ]
    events_target = [
        [(9,11,0), (26,27,0), (33,38,0)],
    ]
    preds, targets = ob.match_bc_events(events_prediction, events_target, threshold_sec=0.0)
    print(preds)
    print(targets)
