import torch
from huggingface_hub import hf_hub_download, list_repo_files

repo_ids = {
    "vap_jp": "maai-kyoto/vap_jp",
    "vap_en": "maai-kyoto/vap_en",
    "vap_ch": "maai-kyoto/vap_ch",
    "vap_tri": "maai-kyoto/vap_tri",
    "vap_jp_kyoto": "maai-kyoto/vap_jp_kyoto",

    "vap_mc_jp": "maai-kyoto/vap_mc_jp",
    "vap_mc_en": "maai-kyoto/vap_mc_en",
    "vap_mc_ch": "maai-kyoto/vap_mc_ch",
    "vap_mc_tri": "maai-kyoto/vap_mc_tri",
    "vap_mc_jp_kyoto": "maai-kyoto/vap_mc_jp_kyoto",

    "vap_bc_jp": "maai-kyoto/vap_bc_jp",

    "vap_bc_2type_jp": "maai-kyoto/vap_bc_2type_jp",
    # "vap_bc_jp_only_timing": "maai-kyoto/vap_bc_jp_only_timing",
    "vap_nod_jp": "maai-kyoto/vap_nod_jp",
    "vap_prompt_jp": "maai-kyoto/vap_prompt_jp",
    # "vap_nod_jp_only_timing": "maai-kyoto/vap_nod_jp_only_timing",
}

def load_vap_model(mode: str, frame_rate: int, context_len_sec: float, language: str = "jp", device: str = "cpu", cache_dir: str = None, force_download: bool = False):
    
    if mode == "vap":
        if language == "jp":
            repo_id = repo_ids["vap_jp"]
            file_path = f"vap_state_dict_jp_{frame_rate}hz_{int(context_len_sec*1000)}msec.pt"
        
        elif language == "en":
            repo_id = repo_ids["vap_en"]
            file_path = f"vap_state_dict_eng_{frame_rate}hz_{int(context_len_sec*1000)}msec.pt"
        
        elif language == "ch":
            repo_id = repo_ids["vap_ch"]
            file_path = f"vap_state_dict_ch_{frame_rate}hz_{int(context_len_sec*1000)}msec.pt"

        elif language == "tri":
            repo_id = repo_ids["vap_tri"]
            file_path = f"vap_state_dict_tri_ecj_{frame_rate}hz_{int(context_len_sec*1000)}msec.pt"

        elif language == "jp_kyoto":
            repo_id = repo_ids["vap_jp_kyoto"]
            file_path = f"vap_state_dict_jp_kyoto_{frame_rate}hz_{int(context_len_sec*1000)}msec.pt"

        else:
            supported_languages = ["jp", "en", "ch", "tri", "jp_kyoto"]
            raise ValueError(f"Invalid language: {language}. Mode {mode} supports languages are: {supported_languages}")

    elif mode == "vap_mc":
        if language == "jp":
            repo_id = repo_ids["vap_mc_jp"]
            file_path = f"vap_mc_state_dict_jp_{frame_rate}hz_{int(context_len_sec*1000)}msec.pt"
        
        elif language == "en":
            repo_id = repo_ids["vap_mc_en"]
            file_path = f"vap_mc_state_dict_eng_{frame_rate}hz_{int(context_len_sec*1000)}msec.pt"
        
        elif language == "ch":
            repo_id = repo_ids["vap_mc_ch"]
            file_path = f"vap_mc_state_dict_ch_{frame_rate}hz_{int(context_len_sec*1000)}msec.pt"
        
        elif language == "tri":
            repo_id = repo_ids["vap_mc_tri"]
            file_path = f"vap_mc_state_dict_tri_{frame_rate}hz_{int(context_len_sec*1000)}msec.pt"
        
        elif language == "jp_kyoto":
            repo_id = repo_ids["vap_mc_jp_kyoto"]
            file_path = f"vap_mc_state_dict_jp_kyoto_{frame_rate}hz_{int(context_len_sec*1000)}msec.pt"

        else:
            supported_languages = ["jp", "en", "ch", "tri", "jp_kyoto"]
            raise ValueError(f"Invalid language: {language}. Mode {mode} supports languages are: {supported_languages}")
    
    elif mode == "bc":
        if language == "jp":
            repo_id = repo_ids["vap_bc_jp"]
            file_path = f"vap-bc_state_dict_jp_{frame_rate}hz_{int(context_len_sec*1000)}msec.pt"
        
        else:
            supported_languages = ["jp"]
            raise ValueError(f"Invalid language: {language}. Mode {mode} supports languages are: {supported_languages}")

    elif mode == "bc_2type":
        
        if language == "jp":
            repo_id = repo_ids["vap_bc_2type_jp"]
            file_path = f"vap-bc-2type_state_dict_jp_{frame_rate}hz_{int(context_len_sec*1000)}msec.pt"

        # elif language == "en":
        #     repo_id = repo_ids["vap_bc_2type_en"]
        #     file_path = f"vap-bc_2type_state_dict_erica_{frame_rate}hz_{int(context_len_sec*1000)}msec.pt"

        else:
            supported_languages = ["jp", "en", "tri"]
            raise ValueError(f"Invalid language: {language}. Mode {mode} supports languages are: {supported_languages}")

    elif mode == "nod":
        
        if language == "jp":
            repo_id = repo_ids["vap_nod_jp"]
            file_path = f"vap-nod_state_dict_erica_{frame_rate}hz_{int(context_len_sec*1000)}msec.pt"
        
        elif language == "en":
            repo_id = repo_ids["vap_nod_en"]
            file_path = f"vap-nod_state_dict_erica_{frame_rate}hz_{int(context_len_sec*1000)}msec.pt"
        
        else:
            supported_languages = ["jp", "en", "tri"]
            raise ValueError(f"Invalid language: {language}. Mode {mode} supports languages are: {supported_languages}")
    
    elif mode == "vap_prompt":

        if language == "jp":
            repo_id = repo_ids["vap_prompt_jp"]
            file_path = f"vap_prompt_state_dict_jp_{frame_rate}hz_{int(context_len_sec*1000)}msec.pt"
        
        else:
            supported_languages = ["jp"]
            raise ValueError(f"Invalid language: {language}. Mode {mode} supports languages are: {supported_languages}")
    else:
        supported_modes = ["vap", "vap_mc", "bc_2type", "nod"]
        raise ValueError(f"Invalid mode: {mode}. Supported modes are: {supported_modes}")

    try:
        sd = hf_hub_download(repo_id=repo_id, filename=file_path, cache_dir=cache_dir, force_download=force_download)
    
    except Exception as e:
        raise ValueError(f"Invalid model: mode: {mode}, frame_rate: {frame_rate}, context_len_sec: {context_len_sec}, language: {language}. Run get_available_models() for available models.")
    
    sd = torch.load(sd, map_location=torch.device(device))
    
    return sd

def get_available_models():
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
