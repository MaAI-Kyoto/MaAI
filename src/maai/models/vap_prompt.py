import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
import torch.nn.functional as F

from .config import VapConfig
from ..encoder import build_audio_encoder
from ..modules import GPT, GPTStereo
from ..objective import ObjectiveVAP

from sentence_transformers import SentenceTransformer

class VapGPT_prompt(nn.Module):
    """Voice Activity Projection with Prompt Control (Beta).
    
    This model integrates text prompts (e.g., personality or instruction prompts)
    into the VAP architecture using sentence embeddings to condition the output.
    """
    BINS_P_NOW = [0, 1]
    BINS_PFUTURE = [2, 3]

    # prompt_model_name = "sbintuitions/sarashina-embedding-v1-1b"
    prompt_model_name = "cl-nagoya/ruri-v3-pt-30m"

    def __init__(self, conf: Optional[VapConfig] = None):
        """Initialize the VapGPT_prompt model.
        
        Args:
            conf (Optional[VapConfig]): Configuration object.
                If None, default VapConfig is used.
        """
        
        # print this model is a beta version
        print('--------------------------------')
        print("<<< This is a beta version of model !!! >>>")
        print("VAP with prompt control is under development.")
        print("This is a beta version of VapGPT with prompt support. It may not work as expected.")
        print("This model also requires 'sentence-transformers protobuf sentencepiece' package to be installed.")
        print('--------------------------------')

        if conf is None:
            conf = VapConfig()
        self.conf = conf
        self.sample_rate = conf.sample_rate
        self.frame_hz = conf.frame_hz

        self.temp_elapse_time = []

        # Single channel
        self.ar_channel = GPT(
            dim=conf.dim,
            dff_k=3,
            num_layers=conf.channel_layers,
            num_heads=conf.num_heads,
            dropout=conf.dropout,
            context_limit=conf.context_limit,
        )

        # Cross channel
        self.ar = GPTStereo(
            dim=conf.dim,
            dff_k=3,
            num_layers=conf.cross_layers,
            num_heads=conf.num_heads,
            dropout=conf.dropout,
            context_limit=conf.context_limit,
        )

        self.objective = ObjectiveVAP(bin_times=conf.bin_times, frame_hz=conf.frame_hz)

        # Outputs
        # Voice activity objective -> x1, x2 -> logits ->  BCE
        self.va_classifier = nn.Linear(conf.dim, 1)
        self.vap_head = nn.Linear(conf.dim, self.objective.n_classes)
        self.prompt_head = nn.Linear(conf.dim, conf.dim_prompt)

        self.prompt_embed1 = nn.Linear(self.conf.dim_prompt, self.conf.dim_prompt_2)
        self.prompt_embed2 = nn.Linear(self.conf.dim_prompt, self.conf.dim_prompt_2)

        self.prompt_dim_red1 = nn.Linear(self.conf.dim + self.conf.dim_prompt_2, self.conf.dim)
        self.prompt_dim_red2 = nn.Linear(self.conf.dim + self.conf.dim_prompt_2, self.conf.dim)

        # Initialize the embedding model for prompts
        print("Loading prompt embedding model:", self.prompt_model_name)
        self.prompt_embedding_model = SentenceTransformer(self.prompt_model_name)
        print("Prompt embedding model loaded.")

        # Initialize the prompt embeddings
        self.set_prompt_ch1("テンポよく発話し、相手の発言が終わるとすぐに返答してください。発言回数を多めに、会話をリードするようにしてください。")
        self.set_prompt_ch2("発話前に少し間を取り、考えてから丁寧に話し始めてください。応答は急がず、落ち着いたテンポを意識してください。")

    def load_encoder(self, cpc_model):
        """Load and build the audio encoders for both speakers.
        
        Args:
            cpc_model: Pre-trained CPC model to be used as feature extractor.
        """
        self.encoder1 = build_audio_encoder(self.conf, cpc_model=cpc_model)
        self.encoder1 = self.encoder1.eval()
        self.encoder2 = build_audio_encoder(self.conf, cpc_model=cpc_model)
        self.encoder2 = self.encoder2.eval()

        encoder_dim = getattr(self.encoder1, "output_dim", self.conf.dim)
        if encoder_dim != self.conf.dim:
            self.decrease_dimension = nn.Linear(encoder_dim, self.conf.dim)
        
        if self.conf.freeze_encoder == 1:
            print('freeze encoder')
            self.encoder1.freeze()
            self.encoder2.freeze()

    @property
    def horizon_time(self):
        """Get the horizon time for the projection in seconds.
        
        Returns:
            float: Horizon time for the objective.
        """
        return self.objective.horizon_time

    def encode_audio(self, audio1: torch.Tensor, audio2: torch.Tensor) -> Tuple[Tensor, Tensor]:
        """Encode the raw audio inputs into feature representations.
        
        Args:
            audio1 (torch.Tensor): Audio waveform for speaker 1.
            audio2 (torch.Tensor): Audio waveform for speaker 2.
            
        Returns:
            Tuple[Tensor, Tensor]: Encoded features for the two speakers.
        """
        
        x1 = self.encoder1(audio1)  # speaker 1
        x2 = self.encoder2(audio2)  # speaker 2

        if hasattr(self, "decrease_dimension"):
            x1 = self.decrease_dimension(x1)
            x2 = self.decrease_dimension(x2)
        
        return x1, x2

    def vad_loss(self, vad_output, vad):
        """Compute the Voice Activity Detection (VAD) loss.
        
        Args:
            vad_output: Predicted VAD logits.
            vad: Ground truth VAD labels.
            
        Returns:
            Tensor: Binary cross-entropy loss between predictions and targets.
        """
        return F.binary_cross_entropy_with_logits(vad_output, vad)
    
    def set_prompt_ch1(self, prompt: str, device: torch.device = torch.device('cpu')):
        """Set the text prompt for channel 1 (speaker 1).
        
        Args:
            prompt (str): The text instruction or personality prompt.
            device (torch.device): The device to load the embedding tensor onto.
        """

        embedding_ch1_ = self.prompt_embedding_model.encode([prompt], normalize_embeddings=True)[0]
        self.embedding_ch1 = torch.tensor(embedding_ch1_).unsqueeze(0)
        self.embedding_ch1 = self.embedding_ch1.to(device=device, non_blocking=True)

        # print("Embedding for channel 1 set:", self.embedding_ch1.shape)
        # print("Embedding for channel 1:", self.embedding_ch1)
        # input("Press Enter to continue...")

    def set_prompt_ch2(self, prompt: str, device: torch.device = torch.device('cpu')):
        """Set the text prompt for channel 2 (speaker 2).
        
        Args:
            prompt (str): The text instruction or personality prompt.
            device (torch.device): The device to load the embedding tensor onto.
        """

        embedding_ch2_ = self.prompt_embedding_model.encode([prompt], normalize_embeddings=True)[0]
        self.embedding_ch2 = torch.tensor(embedding_ch2_).unsqueeze(0)
        self.embedding_ch2 = self.embedding_ch2.to(device=device, non_blocking=True)

        # print("Embedding for channel 2 set:", self.embedding_ch2.shape)
        # print("Embedding for channel 2:", self.embedding_ch2)
        # input("Press Enter to continue...")

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        cache: Optional[dict] = None,
    ) -> Tuple[dict, dict]:
        """
        Forward pass for the VapGPT_prompt model.

        Args:
            x1 (Tensor): Input audio tensor for speaker 1.
            x2 (Tensor): Input audio tensor for speaker 2.
            cache (dict, optional): Cache of past keys/values.

        Returns:
            Tuple[dict, dict]: Output tensors and updated cache.
        """

        if cache is None:
            cache = {}

        prompt_a_seq = self.embedding_ch1.unsqueeze(1).repeat(1, x1.shape[1], 1)
        prompt_b_seq = self.embedding_ch2.unsqueeze(1).repeat(1, x2.shape[1], 1)

        # print("prompt_a_seq", prompt_a_seq.shape)
        # print("prompt_b_seq", prompt_b_seq.shape)
        prompt_a_embed1 = self.prompt_embed1(prompt_a_seq)
        prompt_b_embed1 = self.prompt_embed1(prompt_b_seq)

        # print("prompt_a_embed1", prompt_a_embed1.shape)
        # print("prompt_b_embed1", prompt_b_embed1.shape)

        # print("x1", x1.shape)
        # print("x2", x2.shape)
        
        x1_concat = torch.cat((x1, prompt_a_embed1), dim=-1)
        x2_concat = torch.cat((x2, prompt_b_embed1), dim=-1)
        x1_concat = self.prompt_dim_red1(x1_concat)
        x2_concat = self.prompt_dim_red1(x2_concat)

        o1 = self.ar_channel(x1_concat, past_kv=cache.get("ar1"))
        o2 = self.ar_channel(x2_concat, past_kv=cache.get("ar2"))

        # o1_red = self.prompt_dim_red1(o1["x"])
        # o2_red = self.prompt_dim_red1(o2["x"])

        prompt_a_embed2 = self.prompt_embed2(prompt_a_seq)
        prompt_b_embed2 = self.prompt_embed2(prompt_b_seq)

        o1_concat = torch.cat((o1["x"], prompt_a_embed2), dim=-1)
        o2_concat = torch.cat((o2["x"], prompt_b_embed2), dim=-1)

        o1_concat = self.prompt_dim_red2(o1_concat)
        o2_concat = self.prompt_dim_red2(o2_concat)

        out = self.ar(
            o1_concat,
            o2_concat,
            past_kv1=cache.get("cross1"),
            past_kv2=cache.get("cross2"),
            past_kv1_c=cache.get("cross1_c"),
            past_kv2_c=cache.get("cross2_c"),
        )

        new_cache = {
            "ar1": (o1["past_k"], o1["past_v"]),
            "ar2": (o2["past_k"], o2["past_v"]),
            "cross1": (out["past_k1"], out["past_v1"]),
            "cross2": (out["past_k2"], out["past_v2"]),
            "cross1_c": (out["past_k1_c"], out["past_v1_c"]),
            "cross2_c": (out["past_k2_c"], out["past_v2_c"]),
        }

        # Outputs
        vad1 = self.va_classifier(out["x1"])
        vad2 = self.va_classifier(out["x2"])
        logits = self.vap_head(out["x"])

        # print("logits", logits.shape)
        # print(logits)
        probs = logits.softmax(dim=-1)
                
        p_now = self.objective.probs_next_speaker_aggregate(
            probs,
            from_bin=self.BINS_P_NOW[0],
            to_bin=self.BINS_P_NOW[-1]
        )
        
        p_future = self.objective.probs_next_speaker_aggregate(
            probs,
            from_bin=self.BINS_PFUTURE[0],
            to_bin=self.BINS_PFUTURE[1]
        )
        
        # Get back to the CPU
        p_now = p_now.to('cpu').tolist()[0][-1]
        p_future = p_future.to('cpu').tolist()[0][-1]
        
        vad1 = vad1.sigmoid().to('cpu').tolist()[0][-1][0]
        vad2 = vad2.sigmoid().to('cpu').tolist()[0][-1][0]

        ret = {"p_now": p_now, "p_future": p_future, "vad": [vad1, vad2]}

        return ret, new_cache
