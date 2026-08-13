from dataclasses import dataclass, field
from typing import List

BIN_TIMES: list = [0.2, 0.4, 0.6, 0.8]

@dataclass
class VapConfig:
    """Configuration class for VAP (Voice Activity Projection) model.
    
    This class holds all the hyperparameters and configuration options
    required to build and train the VAP model and its variants.
    """
    sample_rate: int = 16000
    frame_hz: float = 10.0
    bin_times: List[float] = field(default_factory=lambda: BIN_TIMES)

    # Encoder (training flag)
    encoder_type: str = "cpc"
    mimi_model_name: str = "kyutai/mimi"
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

    # --- Nod para（mode=nod_para / MLP 層数・隠れ次元・TaskGPT 層数のみ可変、他は vap_nod_para 内固定）---
    nod_head_mlp_repetitions: int = 1
    nod_head_mlp_range: int = 1
    nod_head_mlp_speed: int = 1
    nod_head_mlp_swing_binary: int = 1
    nod_head_mlp_hidden: int = 128
    nod_task_gpt_layers: int = 2

    # 回数予測の出力形式: 1=binary(1回 vs 2回以上), 0=3クラス
    nod_count_binary: int = 0

    # ===== FiLM (listener-style) conditioning =====
    # 既定はすべて off。film 付き .pt をロードする際に model_config から上書きする。
    nod_film_enable: int = 0
    nod_film_target_ar_channel: int = 0
    nod_film_target_ar_channel_x2: int = 0
    nod_film_target_ar_cross: int = 0
    nod_film_target_ar_cross_x2: int = 0
    nod_film_target_heads: int = 0
    nod_film_heads_scope: str = "all"
    nod_film_cond_dim: int = 0
    nod_film_hidden: int = 0
    nod_film_block_style: str = "post_ffn"
    nod_film_cond_norm: int = 0
    nod_film_cond_zscore: int = 0

    @staticmethod
    def add_argparse_args(parser, fields_added=[]):
        """Add VapConfig attributes as command-line arguments to an argparse parser.
        
        Args:
            parser: The argparse.ArgumentParser instance.
            fields_added (list): A list to keep track of added field names.
            
        Returns:
            tuple: A tuple containing the updated parser and the fields_added list.
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
        """Convert parsed command-line arguments back into a VapConfig instance.
        
        Args:
            args: The parsed arguments from argparse.
            
        Returns:
            VapConfig: A new instance of VapConfig populated with the parsed values.
        """
        return VapConfig(
            **{
                k.replace("vap_", ""): v
                for k, v in vars(args).items()
                if k.startswith("vap_")
            }
        )