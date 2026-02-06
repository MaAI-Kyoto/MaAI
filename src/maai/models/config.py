"""Configuration dataclass for VAP models."""

from dataclasses import dataclass, field
from typing import List

BIN_TIMES: list = [0.2, 0.4, 0.6, 0.8]

@dataclass
class VapConfig:
    """Configuration parameters for VAP model components."""
    sample_rate: int = 16000
    frame_hz: int = 10
    bin_times: List[float] = field(default_factory=lambda: BIN_TIMES)

    # Encoder (training flag)
    encoder_type: str = "cpc"
    wav2vec_type: str = "mms"
    hubert_model: str = "hubert_jp"
    freeze_encoder: int = 1  # stupid but works (--vap_freeze_encoder 1)
    load_pretrained: int = 1  # stupid but works (--vap_load_pretrained 1)
    only_feature_extraction: int = 0

    # GPT
    dim: int = 256
    channel_layers: int = 1
    cross_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1
    context_limit: int = -1

    context_limit_cpc_sec: float = -1

    # Added Multi-task
    lid_classify: int = 0   # 1...last layer, 2...middle layer
    lid_classify_num_class: int = 3
    lid_classify_adversarial: int = 0
    lang_cond: int = 0

    # For prompt
    # dim_prompt: int = 1792
    dim_prompt: int = 256
    dim_prompt_2: int = 256

    @staticmethod
    def add_argparse_args(parser, fields_added=[]):
        """Add configuration fields to an argparse parser.

        Args:
            parser: argparse parser instance.
            fields_added: List to append added field names to.

        Returns:
            Tuple of (parser, fields_added).
        """
        for k, v in VapConfig.__dataclass_fields__.items():
            if k == "bin_times":
                parser.add_argument(
                    f"--vap_{k}", nargs="+", type=float, default=v.default_factory()
                )
            else:
                parser.add_argument(f"--vap_{k}", type=v.type, default=v.default)
            fields_added.append(k)
        return parser, fields_added

    @staticmethod
    def args_to_conf(args):
        """Convert argparse args into a VapConfig instance.

        Args:
            args: argparse namespace with "vap_" prefixed fields.

        Returns:
            VapConfig instance.
        """
        return VapConfig(
            **{
                k.replace("vap_", ""): v
                for k, v in vars(args).items()
                if k.startswith("vap_")
            }
        )
