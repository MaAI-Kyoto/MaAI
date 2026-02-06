"""Audio encoder wrappers and utilities."""

import torch
import torch.nn as nn
import einops
import os
import numpy as np

from .encoder_components import load_CPC, get_cnn_layer

import time

class EncoderCPC(nn.Module):
    """
    Encoder: waveform -> h
    pretrained: default='cpc'

    A simpler version of the Encoder
    check paper (branch) version to see other encoders...
    """

    def __init__(self, load_pretrained=True, freeze=True, cpc_model=''):
        
        super().__init__()
        
        self.sample_rate = 16000
        
        if load_pretrained:
            self.encoder = load_CPC(checkpoint_cpc=cpc_model, load_state_dict=True)
        else:
            self.encoder = load_CPC(checkpoint_cpc='', load_state_dict=False)
        
        # Keep Hidden layer
        self.encoder.gAR.keepHidden = True
        
        self.output_dim = self.encoder.gEncoder.conv4.out_channels
        self.dim = self.output_dim

        self.downsample_ratio = 160
        self.downsample = get_cnn_layer(
            dim=self.output_dim,
            kernel=[5],
            stride=[2],
            dilation=[1],
            activation="GELU",
        )
        self.downsample_ratio = 320

        if freeze:
            self.freeze()

    def get_default_conf(self):
        """Return a placeholder default configuration."""
        return {""}

    def freeze(self):
        """Freeze encoder parameters for inference."""
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        print(f"Froze {self.__class__.__name__}!")

    def unfreeze(self):
        """Unfreeze encoder parameters for training."""
        for p in self.encoder.parameters():
            p.requires_grad_(True)
        print(f"Trainable {self.__class__.__name__}!")

    def forward(self, waveform):
        """Encode waveform frames into downsampled representations.

        Args:
            waveform: Input waveform tensor.

        Returns:
            Encoded feature tensor.
        """
        
        if waveform.ndim < 3:
            waveform = waveform.unsqueeze(1)  # channel dim

        # Backwards using only the encoder encounters:
        # ---------------------------------------------------
        # RuntimeError: one of the variables needed for gradient computation
        # has been modified by an inplace operation:
        # [torch.FloatTensor [4, 256, 1000]], which is output 0 of ReluBackward0, is at version 1;
        # expected version 0 instead. Hint: enable anomaly detection to find
        # the operation that failed to compute its gradient, with
        # torch.autograd.set_detect_anomaly(True).
        # HOWEVER, if we feed through encoder.gAR we do not encounter that problem...

        
        z = self.encoder.gEncoder(waveform)
        z = einops.rearrange(z, "b c n -> b n c")
        z = z[:, 1:-1, :]
        z = self.encoder.gAR(z)
        z = self.downsample(z)
        
        return z

    def hash_tensor(self, tensor):
        """Return a hash of a tensor's flattened values.

        Args:
            tensor: Tensor to hash.

        Returns:
            Integer hash of the flattened tensor values.
        """
        return hash(tuple(tensor.reshape(-1).tolist()))
