import math

import numpy as np
import torch
from huggingface_hub import hf_hub_download, list_repo_files

MODEL_TYPE_TO_ENCODER_TYPE = {
    "normal": "cpc",
    "normal-ver2": "mimi",
}

repo_ids = {
    "vap_jp": "maai-kyoto/vap_jp",
    "vap_en": "maai-kyoto/vap_en",
    "vap_ch": "maai-kyoto/vap_ch",
    "vap_fr": "maai-kyoto/vap_fr",
    "vap_tri": "maai-kyoto/vap_tri",
    "vap_jp_kyoto": "maai-kyoto/vap_jp_kyoto",
    "vap_en_kyoto": "maai-kyoto/vap_en_kyoto",
    "vap_ch_kyoto": "maai-kyoto/vap_ch_kyoto",
    "vap_tri_kyoto": "maai-kyoto/vap_tri_kyoto",

    "vap_ca": "maai-kyoto/vap_ca",

    "vap_mc_jp": "maai-kyoto/vap_mc_jp",
    "vap_mc_en": "maai-kyoto/vap_mc_en",
    "vap_mc_ch": "maai-kyoto/vap_mc_ch",
    "vap_mc_fr": "maai-kyoto/vap_mc_fr",
    "vap_mc_tri": "maai-kyoto/vap_mc_tri",
    "vap_mc_jp_kyoto": "maai-kyoto/vap_mc_jp_kyoto",
    "vap_mc_en_kyoto": "maai-kyoto/vap_mc_en_kyoto",
    "vap_mc_ch_kyoto": "maai-kyoto/vap_mc_ch_kyoto",
    "vap_mc_tri_kyoto": "maai-kyoto/vap_mc_tri_kyoto",

    "vap_bc_jp": "maai-kyoto/vap_bc_jp",
    "vap_bc_en": "maai-kyoto/vap_bc_en",
    "vap_bc_ch": "maai-kyoto/vap_bc_ch",
    "vap_bc_tri": "maai-kyoto/vap_bc_tri",
    "vap_bc_2type_jp": "maai-kyoto/vap_bc_2type_jp",
    # "vap_bc_jp_only_timing": "maai-kyoto/vap_bc_jp_only_timing",
    "vap_nod_jp": "maai-kyoto/vap_nod_jp",
    "vap_nod_para_jp": "maai-kyoto/vap_nod_para_jp",
    "vap_prompt_jp": "maai-kyoto/vap_prompt_jp",
    # "vap_nod_jp_only_timing": "maai-kyoto/vap_nod_jp_only_timing",

    "vad_jp": "maai-kyoto/vad_jp",
    "vad_ch": "maai-kyoto/vad_ch",
    "vad_en": "maai-kyoto/vad_en",

    "bc_det_jp": "maai-kyoto/bc_det_jp",
    "bc_det_en": "maai-kyoto/bc_det_en",
    "bc_det_ch": "maai-kyoto/bc_det_ch",
}

# Streaming Mimi ONNX weights (same hub pattern as VAP checkpoints).
CONTINUOUS_MIMI_ONNX_REPO_ID = "maai-kyoto/continuous-mimi-onnx"

def _format_frame_rate(frame_rate: float) -> str:
    frame_rate = float(frame_rate)
    if frame_rate.is_integer():
        return str(int(frame_rate))
    return f"{frame_rate:g}"


def resolve_encoder_type(model_type: str = "normal") -> str:
    """Resolve the encoder type based on the provided model type.
    
    Args:
        model_type (str): The type of model (e.g., 'normal', 'normal-ver2').
        
    Returns:
        str: The corresponding encoder type ('cpc' or 'mimi').
    """
    try:
        return MODEL_TYPE_TO_ENCODER_TYPE[model_type]
    except KeyError as exc:
        supported_model_types = list(MODEL_TYPE_TO_ENCODER_TYPE.keys())
        raise ValueError(
            f"Unsupported model_type: {model_type}. Supported model_type values are: {supported_model_types}"
        ) from exc


def load_vap_model(mode: str, frame_rate: float, context_len_sec: float, language: str = "jp", device: str = "cpu", cache_dir: str = None, force_download: bool = False, model_type: str = "normal"):
    """Load a pretrained VAP model from the Hugging Face hub.
    
    Args:
        mode (str): The operational mode of the model (e.g., 'vap', 'vap_mc', 'bc', 'nod').
        frame_rate (float): The frame rate expected by the model.
        context_len_sec (float): The context length in seconds.
        language (str): The language identifier for the model (e.g., 'jp', 'en').
        device (str): The device to load the model onto ('cpu', 'cuda').
        cache_dir (str, optional): Directory to cache the downloaded model.
        force_download (bool): If True, forces download even if cached.
        model_type (str): The general model architecture type.
        
    Returns:
        Dict[str, Any]: The loaded state dictionary of the model.
    """
    frame_rate_label = _format_frame_rate(frame_rate)
    encoder_type = resolve_encoder_type(model_type)
    encoder_suffix = ""
    if encoder_type == "mimi":
        encoder_suffix = "_mimi"
    elif encoder_type != "cpc":
        raise ValueError(f"Unsupported encoder_type for pretrained model lookup: {encoder_type}")
    
    if mode == "vap":
        if language == "jp":
            repo_id = repo_ids["vap_jp"]
            file_path = f"vap{encoder_suffix}_state_dict_jp_{frame_rate_label}hz_{int(context_len_sec*1000)}msec.pt"
        
        elif language == "en":
            repo_id = repo_ids["vap_en"]
            file_path = f"vap{encoder_suffix}_state_dict_en_{frame_rate_label}hz_{int(context_len_sec*1000)}msec.pt"
        
        elif language == "ch":
            repo_id = repo_ids["vap_ch"]
            file_path = f"vap{encoder_suffix}_state_dict_ch_{frame_rate_label}hz_{int(context_len_sec*1000)}msec.pt"

        elif language == "tri":
            repo_id = repo_ids["vap_tri"]
            file_path = f"vap{encoder_suffix}_state_dict_tri_ecj_{frame_rate_label}hz_{int(context_len_sec*1000)}msec.pt"

        elif language == "jp_kyoto":
            repo_id = repo_ids["vap_jp_kyoto"]
            file_path = f"vap{encoder_suffix}_state_dict_jp_kyoto_{frame_rate_label}hz_{int(context_len_sec*1000)}msec.pt"
        
        elif language == "en_kyoto":
            repo_id = repo_ids["vap_en_kyoto"]
            file_path = f"vap{encoder_suffix}_state_dict_en_kyoto_{frame_rate_label}hz_{int(context_len_sec*1000)}msec.pt"
        
        elif language == "ch_kyoto":
            repo_id = repo_ids["vap_ch_kyoto"]
            file_path = f"vap{encoder_suffix}_state_dict_ch_kyoto_{frame_rate_label}hz_{int(context_len_sec*1000)}msec.pt"
        
        elif language == "tri_kyoto":
            repo_id = repo_ids["vap_tri_kyoto"]
            file_path = f"vap{encoder_suffix}_state_dict_tri_kyoto_{frame_rate_label}hz_{int(context_len_sec*1000)}msec.pt"
        
        elif language == "ca":
            repo_id = repo_ids["vap_ca"]
            file_path = f"vap{encoder_suffix}_state_dict_ca_{frame_rate_label}hz_{int(context_len_sec*1000)}msec.pt"
        
        elif language == "fr":
            repo_id = repo_ids["vap_fr"]
            file_path = f"vap{encoder_suffix}_state_dict_fr_{frame_rate_label}hz_{int(context_len_sec*1000)}msec.pt"

        else:
            supported_languages = ["jp", "en", "ch", "tri", "jp_kyoto", "en_kyoto", "ch_kyoto", "tri_kyoto", "ca", "fr"]
            raise ValueError(f"Invalid language: {language}. Mode {mode} supports languages are: {supported_languages}")

    # vap_mono is a single-channel model with its own checkpoints
    elif mode == "vap_mono":
        if language in ("jp", "en", "ch"):
            repo_id = repo_ids[f"vap_{language}"]
            file_path = f"vap_mono{encoder_suffix}_state_dict_{language}_{frame_rate_label}hz_{int(context_len_sec*1000)}msec.pt"

        else:
            supported_languages = ["jp", "en", "ch"]
            raise ValueError(f"Invalid language: {language}. Mode {mode} supports languages are: {supported_languages}")

    elif mode == "vap_mc":
        if language == "jp":
            repo_id = repo_ids["vap_mc_jp"]
            file_path = f"vap_mc{encoder_suffix}_state_dict_jp_{frame_rate_label}hz_{int(context_len_sec*1000)}msec.pt"
        
        elif language == "en":
            repo_id = repo_ids["vap_mc_en"]
            file_path = f"vap_mc{encoder_suffix}_state_dict_en_{frame_rate_label}hz_{int(context_len_sec*1000)}msec.pt"
        
        elif language == "ch":
            repo_id = repo_ids["vap_mc_ch"]
            file_path = f"vap_mc{encoder_suffix}_state_dict_ch_{frame_rate_label}hz_{int(context_len_sec*1000)}msec.pt"
        
        elif language == "tri":
            repo_id = repo_ids["vap_mc_tri"]
            file_path = f"vap_mc{encoder_suffix}_state_dict_tri_{frame_rate_label}hz_{int(context_len_sec*1000)}msec.pt"
        
        elif language == "fr":
            repo_id = repo_ids["vap_mc_fr"]
            file_path = f"vap_mc{encoder_suffix}_state_dict_fr_{frame_rate_label}hz_{int(context_len_sec*1000)}msec.pt"
        
        elif language == "jp_kyoto":
            repo_id = repo_ids["vap_mc_jp_kyoto"]
            file_path = f"vap_mc{encoder_suffix}_state_dict_jp_kyoto_{frame_rate_label}hz_{int(context_len_sec*1000)}msec.pt"
        
        elif language == "en_kyoto":
            repo_id = repo_ids["vap_mc_en_kyoto"]
            file_path = f"vap_mc{encoder_suffix}_state_dict_en_kyoto_{frame_rate_label}hz_{int(context_len_sec*1000)}msec.pt"
        
        elif language == "ch_kyoto":
            repo_id = repo_ids["vap_mc_ch_kyoto"]
            file_path = f"vap_mc{encoder_suffix}_state_dict_ch_kyoto_{frame_rate_label}hz_{int(context_len_sec*1000)}msec.pt"
        
        elif language == "tri_kyoto":
            repo_id = repo_ids["vap_mc_tri_kyoto"]
            file_path = f"vap_mc{encoder_suffix}_state_dict_tri_kyoto_{frame_rate_label}hz_{int(context_len_sec*1000)}msec.pt"

        else:
            supported_languages = ["jp", "en", "ch", "tri", "jp_kyoto", "en_kyoto", "ch_kyoto", "tri_kyoto", "fr"]
            raise ValueError(f"Invalid language: {language}. Mode {mode} supports languages are: {supported_languages}")
    
    elif mode == "vad":
        if language == "jp":
            repo_id = repo_ids["vad_jp"]
            file_path = f"vad{encoder_suffix}_state_dict_jp_{frame_rate_label}hz_{int(context_len_sec*1000)}msec.pt"

        elif language == "ch":
            repo_id = repo_ids["vad_ch"]
            file_path = f"vad{encoder_suffix}_state_dict_ch_{frame_rate_label}hz_{int(context_len_sec*1000)}msec.pt"

        elif language == "en":
            repo_id = repo_ids["vad_en"]
            file_path = f"vad{encoder_suffix}_state_dict_en_{frame_rate_label}hz_{int(context_len_sec*1000)}msec.pt"

        else:
            supported_languages = ["jp", "ch", "en"]
            raise ValueError(f"Invalid language: {language}. Mode {mode} supports languages are: {supported_languages}")

    elif mode == "vad_mono":
        if language == "jp":
            repo_id = repo_ids["vad_jp"]
            file_path = f"vad_mono{encoder_suffix}_state_dict_jp_{frame_rate_label}hz_{int(context_len_sec*1000)}msec.pt"

        elif language == "ch":
            repo_id = repo_ids["vad_ch"]
            file_path = f"vad_mono{encoder_suffix}_state_dict_ch_{frame_rate_label}hz_{int(context_len_sec*1000)}msec.pt"

        elif language == "en":
            repo_id = repo_ids["vad_en"]
            file_path = f"vad_mono{encoder_suffix}_state_dict_en_{frame_rate_label}hz_{int(context_len_sec*1000)}msec.pt"

        else:
            supported_languages = ["jp", "ch", "en"]
            raise ValueError(f"Invalid language: {language}. Mode {mode} supports languages are: {supported_languages}")

    elif mode == "bc_det":
        if language in ("jp", "en", "ch"):
            repo_id = repo_ids[f"bc_det_{language}"]
            file_path = f"bc_det{encoder_suffix}_state_dict_{language}_{frame_rate_label}hz_{int(context_len_sec*1000)}msec.pt"

        else:
            supported_languages = ["jp", "en", "ch"]
            raise ValueError(f"Invalid language: {language}. Mode {mode} supports languages are: {supported_languages}")

    elif mode == "bc_det_mono":
        if language in ("jp", "en", "ch"):
            repo_id = repo_ids[f"bc_det_{language}"]
            file_path = f"bc_det_mono{encoder_suffix}_state_dict_{language}_{frame_rate_label}hz_{int(context_len_sec*1000)}msec.pt"

        else:
            supported_languages = ["jp", "en", "ch"]
            raise ValueError(f"Invalid language: {language}. Mode {mode} supports languages are: {supported_languages}")

    elif mode == "bc":
        if language == "jp":
            repo_id = repo_ids["vap_bc_jp"]
            file_path = f"vap-bc{encoder_suffix}_state_dict_jp_{frame_rate_label}hz_{int(context_len_sec*1000)}msec.pt"

        elif language == "en":
            repo_id = repo_ids["vap_bc_en"]
            file_path = f"vap-bc{encoder_suffix}_state_dict_en_{frame_rate_label}hz_{int(context_len_sec*1000)}msec.pt"
        
        elif language == "ch":
            repo_id = repo_ids["vap_bc_ch"]
            file_path = f"vap-bc{encoder_suffix}_state_dict_ch_{frame_rate_label}hz_{int(context_len_sec*1000)}msec.pt"
        
        elif language == "tri":
            repo_id = repo_ids["vap_bc_tri"]
            file_path = f"vap-bc{encoder_suffix}_state_dict_tri_{frame_rate_label}hz_{int(context_len_sec*1000)}msec.pt"
        
        else:
            supported_languages = ["jp", "en", "ch", "tri"]
            raise ValueError(f"Invalid language: {language}. Mode {mode} supports languages are: {supported_languages}")
    
    elif mode == "bc_2type":
        
        if language == "jp":
            repo_id = repo_ids["vap_bc_2type_jp"]
            file_path = f"vap-bc-2type{encoder_suffix}_state_dict_jp_{frame_rate_label}hz_{int(context_len_sec*1000)}msec.pt"

        # elif language == "en":
        #     repo_id = repo_ids["vap_bc_2type_en"]
        #     file_path = f"vap-bc_2type_state_dict_erica_{frame_rate}hz_{int(context_len_sec*1000)}msec.pt"

        else:
            supported_languages = ["jp", "en", "tri"]
            raise ValueError(f"Invalid language: {language}. Mode {mode} supports languages are: {supported_languages}")

    elif mode == "nod":
        
        if language == "jp":
            repo_id = repo_ids["vap_nod_jp"]
            file_path = f"vap-nod{encoder_suffix}_state_dict_erica_{frame_rate_label}hz_{int(context_len_sec*1000)}msec.pt"
        
        elif language == "en":
            repo_id = repo_ids["vap_nod_en"]
            file_path = f"vap-nod{encoder_suffix}_state_dict_erica_{frame_rate_label}hz_{int(context_len_sec*1000)}msec.pt"
        
        else:
            supported_languages = ["jp", "en", "tri"]
            raise ValueError(f"Invalid language: {language}. Mode {mode} supports languages are: {supported_languages}")
    
    elif mode == "vap_prompt":

        if language == "jp":
            repo_id = repo_ids["vap_prompt_jp"]
            file_path = f"vap_prompt{encoder_suffix}_state_dict_jp_{frame_rate_label}hz_{int(context_len_sec*1000)}msec.pt"
        
        else:
            supported_languages = ["jp"]
            raise ValueError(f"Invalid language: {language}. Mode {mode} supports languages are: {supported_languages}")

    elif mode == "nod_para":
        if language != "jp":
            supported_languages = ["jp"]
            raise ValueError(
                f"Invalid language: {language}. Mode {mode} supports languages are: {supported_languages}"
            )
        repo_id = repo_ids["vap_nod_para_jp"]
        file_path = (
            f"vap-nod_para_state_dict_erica_{frame_rate_label}hz_{int(context_len_sec*1000)}msec.pt"
        )

    else:
        supported_modes = ["vap", "vap_mono", "vap_mc", "vad", "vad_mono", "bc_det", "bc_det_mono", "bc", "bc_2type", "nod", "vap_prompt", "nod_para"]
        raise ValueError(f"Invalid mode: {mode}. Supported modes are: {supported_modes}")

    try:
        sd = hf_hub_download(repo_id=repo_id, filename=file_path, cache_dir=cache_dir, force_download=force_download)
    
    except Exception as e:
        raise ValueError(f"Invalid model: mode: {mode}, frame_rate: {frame_rate}, context_len_sec: {context_len_sec}, language: {language}. Run get_available_models() for available models.")
    
    sd = torch.load(sd, map_location=torch.device(device))
    
    return sd

def get_available_models():
    """Retrieve a dictionary of available pre-trained models from the Hugging Face hub.
    
    Returns:
        Dict[str, List[str]]: A mapping of repository IDs to their available model files.
    """
    available_models = {}
    for repo_id in repo_ids.values():
        files = list_repo_files(repo_id)
        available_models[repo_id] = [file for file in files if file.endswith(".pt")]
    return available_models

import struct

BYTE_ORDER = 'little'

#
# Int16 -> Byte
#

def conv_2int16_2_byte(val1, val2):
    """Convert two integers into a combined byte array using 2 bytes each.
    
    Args:
        val1 (int): First integer.
        val2 (int): Second integer.
        
    Returns:
        bytes: Combined byte array.
    """
    b1 = val1.to_bytes(2, BYTE_ORDER)
    b2 = val2.to_bytes(2, BYTE_ORDER)
    
    # print(b1)
    # print(b2)
    # concatenate two bytes
    b = b1 + b2
    
    #print(b)
    
    return b

def conv_2int16array_2_bytearray(arr1, arr2):
    
    if len(arr1) != len(arr2):
        raise ValueError('Two arrays must have the same length')
    
    b = b''
    
    for i in range(len(arr1)):
        b += conv_2int16_2_byte(int(arr1[i]), int(arr2[i]))
    
    return b

#
# Float32 -> Byte
#

def conv_2float_2_byte(val1, val2):
    """Convert two double-precision floats into a combined byte array.
    
    Args:
        val1 (float): First float.
        val2 (float): Second float.
        
    Returns:
        bytes: Combined byte array.
    """
    b1 = struct.pack('<d', val1)
    b2 = struct.pack('<d', val2)
    
    b = b1 + b2
    
    return b

def conv_2floatarray_2_bytearray(arr1, arr2):
    
    if len(arr1) != len(arr2):
        raise ValueError('Two arrays must have the same length')
    
    b = b''
    
    for i in range(len(arr1)):
        b += conv_2float_2_byte(arr1[i], arr2[i])
    
    return b

def conv_float32_2_byte(val1, val2):
    
    b1 = struct.pack('<d', val1)
    b2 = struct.pack('<d', val2)

    b = b1 + b2
    
    return b

def conv_floatarray_2_byte(arr):
    
    b = b''
    
    for i in range(len(arr)):
        b += struct.pack('<d', arr[i])
    
    return b

#
# Byte -> Float32
#

def conv_byte_2_2float(b1, b2):
    """Convert two double-precision byte blocks back into floats.
    
    Args:
        b1 (bytes): First byte block.
        b2 (bytes): Second byte block.
        
    Returns:
        Tuple[float, float]: The decoded floats.
    """
    val1 = struct.unpack('<d', b1)[0]
    val2 = struct.unpack('<d', b2)[0]
    
    return val1, val2

def conv_bytearray_2_2floatarray(barr):
    
    arr1, arr2 = [], []
    
    for i in range(0, len(barr), 16):
        b1 = barr[i:i+8]
        b2 = barr[i+8:i+16]
        
        val1, val2 = conv_byte_2_2float(b1, b2)
        
        arr1.append(val1)
        arr2.append(val2)
        
    return arr1, arr2

def conv_bytearray_2_floatarray(barr):
    
    arr = []
    
    for i in range(0, len(barr), 8):
        b = barr[i:i+8]
        val = struct.unpack('<d', b)[0]
        arr.append(val)
        
    return arr

def conv_bytearray_2_floatarray_short(barr):
    arr = []
    for i in range(0, len(barr), 4):
        b = barr[i:i+4]
        val = struct.unpack('<f', b)[0]
        arr.append(val)
    return arr

#
# VAP result -> Byte
#
def conv_vapresult_2_bytearray(vap_result):
    """Serialize a VAP result dictionary into a byte array.
    
    Args:
        vap_result (Dict[str, Any]): VAP result data.
        
    Returns:
        bytes: The serialized byte array.
    """
    b = b''
    #print(type(vap_result['t']))
    b += struct.pack('<d', vap_result['t'])
    
    b += len(vap_result['x1']).to_bytes(4, BYTE_ORDER)
    b += conv_floatarray_2_byte(vap_result['x1'])
    
    b += len(vap_result['x2']).to_bytes(4, BYTE_ORDER)
    b += conv_floatarray_2_byte(vap_result['x2'])
    
    b += len(vap_result['p_now']).to_bytes(4, BYTE_ORDER)
    b += conv_floatarray_2_byte(vap_result['p_now'])

    b += len(vap_result['p_future']).to_bytes(4, BYTE_ORDER)
    b += conv_floatarray_2_byte(vap_result['p_future'])
    
    b += len(vap_result['vad']).to_bytes(4, BYTE_ORDER)
    b += conv_floatarray_2_byte(vap_result['vad'])
    
    return b

#
# Byte -> VAP result
#
def conv_bytearray_2_vapresult(barr):
    """Deserialize a byte array back into a VAP result dictionary.
    
    Args:
        barr (bytes): Serialized byte array.
        
    Returns:
        Dict[str, Any]: The decoded VAP result data.
    """
    idx = 0
    t = struct.unpack('<d', barr[idx:8])[0]
    idx += 8
    
    len_x1 = struct.unpack('<I', barr[idx:idx+4])[0]
    idx += 4
    x1 = conv_bytearray_2_floatarray(barr[idx:idx+8*len_x1])
    idx += 8*len_x1
    
    len_x2 = struct.unpack('<I', barr[idx:idx+4])[0]
    idx += 4
    x2 = conv_bytearray_2_floatarray(barr[idx:idx+8*len_x2])
    idx += 8 * len_x2
    
    len_p_now = struct.unpack('<I', barr[idx:idx+4])[0]
    idx += 4
    p_now = conv_bytearray_2_floatarray(barr[idx:idx+8*len_p_now])
    idx += 8*len_p_now
    
    len_p_future = struct.unpack('<I', barr[idx:idx+4])[0]
    idx += 4
    p_future = conv_bytearray_2_floatarray(barr[idx:idx+8*len_p_future])
    idx += 8*len_p_future
    
    len_vad = struct.unpack('<I', barr[idx:idx+4])[0]
    idx += 4
    vad = conv_bytearray_2_floatarray(barr[idx:idx+8*len_vad])
    idx += 8*len_vad

    result_vap = {
        't': t,
        'x1': x1,
        'x2': x2,
        'p_now': p_now,
        'p_future': p_future,
        'vad': vad
    }

    return result_vap

#
# VAP result (mono) -> Byte
#
def conv_vapresult_2_bytearray_mono(vap_result):
    """Serialize a mono VAP result dictionary into a byte array.

    Scalar ``p_now`` / ``p_future`` / ``vad`` values are wrapped into
    length-1 arrays so the wire format stays identical to the stereo one.

    Args:
        vap_result (Dict[str, Any]): Mono VAP result data.

    Returns:
        bytes: The serialized byte array.
    """
    wrapped = dict(vap_result)
    wrapped['p_now'] = [vap_result['p_now']]
    wrapped['p_future'] = [vap_result['p_future']]
    wrapped['vad'] = [vap_result['vad']]
    return conv_vapresult_2_bytearray(wrapped)

#
# Byte -> VAP result (mono)
#
def conv_bytearray_2_vapresult_mono(barr):
    """Deserialize a byte array back into a mono VAP result dictionary.

    Args:
        barr (bytes): Serialized byte array.

    Returns:
        Dict[str, Any]: The decoded mono VAP result data with scalar
        ``p_now`` / ``p_future`` / ``vad``.
    """
    result_vap = conv_bytearray_2_vapresult(barr)
    for k in ('p_now', 'p_future', 'vad'):
        result_vap[k] = result_vap[k][0]
    return result_vap

#
# VAD result -> Byte
#
def conv_vapresult_2_bytearray_vad(vap_result):
    """Serialize a VAD result dictionary into a byte array.

    Args:
        vap_result (Dict[str, Any]): VAD result data.

    Returns:
        bytes: The serialized byte array.
    """
    b = b''
    b += struct.pack('<d', vap_result['t'])

    b += len(vap_result['x1']).to_bytes(4, BYTE_ORDER)
    b += conv_floatarray_2_byte(vap_result['x1'])

    b += len(vap_result['x2']).to_bytes(4, BYTE_ORDER)
    b += conv_floatarray_2_byte(vap_result['x2'])

    b += len(vap_result['vad']).to_bytes(4, BYTE_ORDER)
    b += conv_floatarray_2_byte(vap_result['vad'])

    return b

#
# Byte -> VAD result
#
def conv_bytearray_2_vapresult_vad(barr):
    """Deserialize a byte array back into a VAD result dictionary.

    Args:
        barr (bytes): Serialized byte array.

    Returns:
        Dict[str, Any]: The decoded VAD result data.
    """
    idx = 0
    t = struct.unpack('<d', barr[idx:8])[0]
    idx += 8

    len_x1 = struct.unpack('<I', barr[idx:idx+4])[0]
    idx += 4
    x1 = conv_bytearray_2_floatarray(barr[idx:idx+8*len_x1])
    idx += 8*len_x1

    len_x2 = struct.unpack('<I', barr[idx:idx+4])[0]
    idx += 4
    x2 = conv_bytearray_2_floatarray(barr[idx:idx+8*len_x2])
    idx += 8*len_x2

    len_vad = struct.unpack('<I', barr[idx:idx+4])[0]
    idx += 4
    vad = conv_bytearray_2_floatarray(barr[idx:idx+8*len_vad])
    idx += 8*len_vad

    result_vap = {
        't': t,
        'x1': x1,
        'x2': x2,
        'vad': vad
    }

    return result_vap

#
# VAD result (mono) -> Byte
#
def conv_vapresult_2_bytearray_vad_mono(vap_result):
    """Serialize a mono VAD result dictionary into a byte array.

    The scalar ``vad`` value is wrapped into a length-1 array so the wire
    format stays identical to the two-channel one.

    Args:
        vap_result (Dict[str, Any]): Mono VAD result data.

    Returns:
        bytes: The serialized byte array.
    """
    wrapped = dict(vap_result)
    wrapped['vad'] = [vap_result['vad']]
    return conv_vapresult_2_bytearray_vad(wrapped)

#
# Byte -> VAD result (mono)
#
def conv_bytearray_2_vapresult_vad_mono(barr):
    """Deserialize a byte array back into a mono VAD result dictionary.

    Args:
        barr (bytes): Serialized byte array.

    Returns:
        Dict[str, Any]: The decoded mono VAD result data with a scalar ``vad``.
    """
    result_vap = conv_bytearray_2_vapresult_vad(barr)
    result_vap['vad'] = result_vap['vad'][0]
    return result_vap

#
# BC detection result -> Byte
#
def conv_vapresult_2_bytearray_bc_det(vap_result):
    """Serialize a BC detection result dictionary into a byte array.

    Args:
        vap_result (Dict[str, Any]): BC detection result data.

    Returns:
        bytes: The serialized byte array.
    """
    b = b''
    b += struct.pack('<d', vap_result['t'])

    b += len(vap_result['x1']).to_bytes(4, BYTE_ORDER)
    b += conv_floatarray_2_byte(vap_result['x1'])

    b += len(vap_result['x2']).to_bytes(4, BYTE_ORDER)
    b += conv_floatarray_2_byte(vap_result['x2'])

    b += len(vap_result['p_bc_det']).to_bytes(4, BYTE_ORDER)
    b += conv_floatarray_2_byte(vap_result['p_bc_det'])

    return b

#
# Byte -> BC detection result
#
def conv_bytearray_2_vapresult_bc_det(barr):
    """Deserialize a byte array back into a BC detection result dictionary.

    Args:
        barr (bytes): Serialized byte array.

    Returns:
        Dict[str, Any]: The decoded BC detection result data.
    """
    idx = 0
    t = struct.unpack('<d', barr[idx:8])[0]
    idx += 8

    len_x1 = struct.unpack('<I', barr[idx:idx+4])[0]
    idx += 4
    x1 = conv_bytearray_2_floatarray(barr[idx:idx+8*len_x1])
    idx += 8*len_x1

    len_x2 = struct.unpack('<I', barr[idx:idx+4])[0]
    idx += 4
    x2 = conv_bytearray_2_floatarray(barr[idx:idx+8*len_x2])
    idx += 8*len_x2

    len_bc_det = struct.unpack('<I', barr[idx:idx+4])[0]
    idx += 4
    p_bc_det = conv_bytearray_2_floatarray(barr[idx:idx+8*len_bc_det])
    idx += 8*len_bc_det

    result_vap = {
        't': t,
        'x1': x1,
        'x2': x2,
        'p_bc_det': p_bc_det
    }

    return result_vap

#
# BC detection result (mono) -> Byte
#
def conv_vapresult_2_bytearray_bc_det_mono(vap_result):
    """Serialize a mono BC detection result dictionary into a byte array.

    The scalar ``p_bc_det`` value is wrapped into a length-1 array so the
    wire format stays identical to the two-channel one.

    Args:
        vap_result (Dict[str, Any]): Mono BC detection result data.

    Returns:
        bytes: The serialized byte array.
    """
    wrapped = dict(vap_result)
    wrapped['p_bc_det'] = [vap_result['p_bc_det']]
    return conv_vapresult_2_bytearray_bc_det(wrapped)

#
# Byte -> BC detection result (mono)
#
def conv_bytearray_2_vapresult_bc_det_mono(barr):
    """Deserialize a byte array back into a mono BC detection result dictionary.

    Args:
        barr (bytes): Serialized byte array.

    Returns:
        Dict[str, Any]: The decoded mono BC detection result data with a
        scalar ``p_bc_det``.
    """
    result_vap = conv_bytearray_2_vapresult_bc_det(barr)
    result_vap['p_bc_det'] = result_vap['p_bc_det'][0]
    return result_vap

#
# VAP result -> Byte
#
def conv_vapresult_2_bytearray_bc_2type(vap_result):
    
    b = b''
    #print(type(vap_result['t']))
    b += struct.pack('<d', vap_result['t'])
    
    b += len(vap_result['x1']).to_bytes(4, BYTE_ORDER)
    b += conv_floatarray_2_byte(vap_result['x1'])
    
    b += len(vap_result['x2']).to_bytes(4, BYTE_ORDER)
    b += conv_floatarray_2_byte(vap_result['x2'])
    
    b += len(vap_result['p_bc_react']).to_bytes(4, BYTE_ORDER)
    b += conv_floatarray_2_byte(vap_result['p_bc_react'])

    b += len(vap_result['p_bc_emo']).to_bytes(4, BYTE_ORDER)
    b += conv_floatarray_2_byte(vap_result['p_bc_emo'])
    
    return b

def conv_vapresult_2_bytearray_nod(vap_result):
    
    b = b''
    #print(type(vap_result['t']))
    b += struct.pack('<d', vap_result['t'])
    
    b += len(vap_result['x1']).to_bytes(4, BYTE_ORDER)
    b += conv_floatarray_2_byte(vap_result['x1'])
    
    b += len(vap_result['x2']).to_bytes(4, BYTE_ORDER)
    b += conv_floatarray_2_byte(vap_result['x2'])
    
    b += len(vap_result['p_bc']).to_bytes(4, BYTE_ORDER)
    b += conv_floatarray_2_byte(vap_result['p_bc'])

    b += len(vap_result['p_nod_short']).to_bytes(4, BYTE_ORDER)
    b += conv_floatarray_2_byte(vap_result['p_nod_short'])
    
    b += len(vap_result['p_nod_long']).to_bytes(4, BYTE_ORDER)
    b += conv_floatarray_2_byte(vap_result['p_nod_long'])
    
    b += len(vap_result['p_nod_long_p']).to_bytes(4, BYTE_ORDER)
    b += conv_floatarray_2_byte(vap_result['p_nod_long_p'])
    
    return b

#
# Byte -> VAP result
#
def conv_bytearray_2_vapresult_bc_2type(barr):
    
    idx = 0
    t = struct.unpack('<d', barr[idx:8])[0]
    idx += 8
    
    len_x1 = struct.unpack('<I', barr[idx:idx+4])[0]
    idx += 4
    x1 = conv_bytearray_2_floatarray(barr[idx:idx+8*len_x1])
    idx += 8*len_x1
    
    len_x2 = struct.unpack('<I', barr[idx:idx+4])[0]
    idx += 4
    x2 = conv_bytearray_2_floatarray(barr[idx:idx+8*len_x2])
    idx += 8 * len_x2
    
    len_p_bc_react = struct.unpack('<I', barr[idx:idx+4])[0]
    idx += 4
    p_bc_react = conv_bytearray_2_floatarray(barr[idx:idx+8*len_p_bc_react])
    idx += 8*len_p_bc_react
    
    len_p_bc_emo = struct.unpack('<I', barr[idx:idx+4])[0]
    idx += 4
    p_bc_emo = conv_bytearray_2_floatarray(barr[idx:idx+8*len_p_bc_emo])

    result_vap = {
        't': t,
        'x1': x1,
        'x2': x2,
        'p_bc_react': p_bc_react,
        'p_bc_emo': p_bc_emo
    }
    
    return result_vap

def conv_bytearray_2_vapresult_nod(barr):
    
    idx = 0
    t = struct.unpack('<d', barr[idx:8])[0]
    idx += 8
    
    len_x1 = struct.unpack('<I', barr[idx:idx+4])[0]
    idx += 4
    x1 = conv_bytearray_2_floatarray(barr[idx:idx+8*len_x1])
    idx += 8*len_x1
    
    len_x2 = struct.unpack('<I', barr[idx:idx+4])[0]
    idx += 4
    x2 = conv_bytearray_2_floatarray(barr[idx:idx+8*len_x2])
    idx += 8*len_x2
    
    len_p_bc = struct.unpack('<I', barr[idx:idx+4])[0]
    idx += 4
    p_bc = conv_bytearray_2_floatarray(barr[idx:idx+8*len_p_bc])
    idx += 8*len_p_bc
    
    len_p_nod_short = struct.unpack('<I', barr[idx:idx+4])[0]
    idx += 4
    p_nod_short = conv_bytearray_2_floatarray(barr[idx:idx+8*len_p_nod_short])
    idx += 8*len_p_nod_short
    
    len_p_nod_long = struct.unpack('<I', barr[idx:idx+4])[0]
    idx += 4
    p_nod_long = conv_bytearray_2_floatarray(barr[idx:idx+8*len_p_nod_long])
    idx += 8*len_p_nod_long
    
    len_p_nod_long_p = struct.unpack('<I', barr[idx:idx+4])[0]
    idx += 4
    p_nod_long_p = conv_bytearray_2_floatarray(barr[idx:idx+8*len_p_nod_long_p])

    result_vap = {
        't': t,
        'x1': x1,
        'x2': x2,
        'p_bc': p_bc,
        'p_nod_short': p_nod_short,
        'p_nod_long': p_nod_long,
        'p_nod_long_p': p_nod_long_p
    }
    
    return result_vap

def download_continuous_mimi_onnx(
    precision: str = "fp32",
    cache_dir: str | None = None,
    force_download: bool = False,
    frames_per_call: int = 1,
) -> tuple[str, str]:
    """
    Resolve paths to the streaming Mimi ONNX model and JSON sidecar on disk.

    Files are fetched from ``maai-kyoto/continuous-mimi-onnx`` via ``hf_hub_download``
    (cached under the usual Hugging Face cache layout, or under ``cache_dir`` when set).

    ``frames_per_call`` selects the microbatch export contract: 1 (default) is the
    original one-frame-per-call export (``continuous_mimi_{precision}.onnx``); N > 1
    selects the N-frames-per-call export (``continuous_mimi_{precision}_{N}f.onnx``),
    which produces N VAP frames per ONNX call.
    """
    precision = str(precision).strip().lower()
    frames_per_call = int(frames_per_call)
    if frames_per_call < 1:
        raise ValueError("frames_per_call must be at least 1.")
    suffix = "" if frames_per_call == 1 else f"_{frames_per_call}f"
    if precision == "fp32":
        onnx_fn = f"continuous_mimi_fp32{suffix}.onnx"
        meta_fn = f"continuous_mimi_fp32{suffix}.json"
    elif precision == "int8":
        onnx_fn = f"continuous_mimi_int8{suffix}.onnx"
        meta_fn = f"continuous_mimi_int8{suffix}.json"
    else:
        raise ValueError(f"Unsupported precision for continuous Mimi ONNX: {precision}")

    onnx_path = hf_hub_download(
        repo_id=CONTINUOUS_MIMI_ONNX_REPO_ID,
        filename=onnx_fn,
        cache_dir=cache_dir,
        force_download=force_download,
    )
    meta_path = hf_hub_download(
        repo_id=CONTINUOUS_MIMI_ONNX_REPO_ID,
        filename=meta_fn,
        cache_dir=cache_dir,
        force_download=force_download,
    )
    return str(onnx_path), str(meta_path)

def generate_natural_nod(
    range_rad: float,
    count: int,
    use_pre_rise: bool,
    velocity: float,
    fps: int = 30,
    decay_rate: float = 0.6,
    pre_rise_ratio: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a natural nodding motion sequence (pitch in radians vs time in seconds).

    Interpolation is cubic spline (``CubicSpline``) when scipy is available,
    else cosine interpolation between keyframes.

    Parameters
    ----------
    range_rad : float
        Nod depth in radians (absolute value).
    count : int
        Number of nods (>= 1).
    use_pre_rise : bool
        Whether to include a pre-rise before the first nod.
    velocity : float
        Target average angular velocity (rad/s).
    fps : int
        Output frame rate.
    decay_rate : float
        Amplitude decay per nod (0–1).
    pre_rise_ratio : float
        Pre-rise amplitude as a ratio of *range_rad*.

    Returns
    -------
    motion : np.ndarray
        Pitch values in radians, one per frame.
    time_axis : np.ndarray
        Time stamps in seconds, same length as *motion*.
    """
    count = max(1, int(count))
    fps = max(1, int(fps))

    keyframe_vals: list[float] = [0.0]
    current_amp = abs(float(range_rad))

    if use_pre_rise:
        keyframe_vals.extend(
            [
                current_amp * float(pre_rise_ratio),
                -current_amp,
                0.0,
            ]
        )
    else:
        keyframe_vals.extend([-current_amp, 0.0])

    for _ in range(count - 1):
        current_amp *= float(decay_rate)
        keyframe_vals.extend([-current_amp, 0.0])

    n_seg = len(keyframe_vals) - 1
    distances: list[float] = []
    vel_scales: list[float] = []
    for i in range(n_seg):
        d = abs(keyframe_vals[i + 1] - keyframe_vals[i])
        distances.append(d)
        if d < 1e-6:
            vel_scales.append(1.0)
        else:
            if use_pre_rise:
                nod_idx = 0 if i < 3 else 1 + (i - 3) // 2
            else:
                nod_idx = i // 2
            vel_scales.append(float(decay_rate) ** (nod_idx * 0.5))

    total_distance = sum(distances)
    raw_total = sum(d / vs for d, vs in zip(distances, vel_scales))
    total_time = total_distance / float(velocity) if float(velocity) > 1e-9 else 0.0

    keyframe_frames: list[int] = [0]
    for d, vs in zip(distances, vel_scales):
        if raw_total > 1e-9 and d >= 1e-6:
            duration = (d / vs) * total_time / raw_total
            n_frames = max(1, int(round(duration * fps)))
        else:
            n_frames = 0
        keyframe_frames.append(keyframe_frames[-1] + n_frames)

    keyframe_times = [f / fps for f in keyframe_frames]
    total_frames = keyframe_frames[-1]

    if total_frames <= 0:
        return np.array([0.0]), np.array([0.0])

    time_axis = np.arange(total_frames) / fps

    motion = _nod_motion_cubic_spline_or_cosine(
        keyframe_times,
        keyframe_vals,
        keyframe_frames,
        time_axis,
    )
    return motion, time_axis

def _nod_motion_cubic_spline_or_cosine(
    keyframe_times: list[float],
    keyframe_vals: list[float],
    keyframe_frames: list[int],
    time_axis: np.ndarray,
) -> np.ndarray:
    try:
        from scipy.interpolate import CubicSpline

        kt = np.asarray(keyframe_times, dtype=float)
        if kt.size >= 2 and np.all(np.diff(kt) > 1e-15):
            kv = np.asarray(keyframe_vals, dtype=float)
            cs = CubicSpline(kt, kv, bc_type=((2, 0), (1, 0)))
            return np.asarray(cs(time_axis), dtype=float)
    except ImportError:
        raise ImportError("scipy is not installed")
    except Exception as e:
        raise Exception(f"Error in _nod_motion_cubic_spline_or_cosine: {e}")

def euler_to_quaternion(rx: float, ry: float, rz: float) -> tuple[float, float, float, float]:
    """XYZ intrinsic Euler angles (radians) to quaternion (qx, qy, qz, qw)."""
    cx = math.cos(rx / 2.0)
    sx = math.sin(rx / 2.0)
    cy = math.cos(ry / 2.0)
    sy = math.sin(ry / 2.0)
    cz = math.cos(rz / 2.0)
    sz = math.sin(rz / 2.0)
    qx = sx * cy * cz + cx * sy * sz
    qy = cx * sy * cz - sx * cy * sz
    qz = cx * cy * sz + sx * sy * cz
    qw = cx * cy * cz - sx * sy * sz
    return (qx, qy, qz, qw)