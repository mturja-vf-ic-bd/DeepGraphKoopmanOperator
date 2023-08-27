import re


def parse_model_name(model_name):
    inpd_ptn = '_dim\d+'
    inpl_ptn = '_inp\d+'
    step_ptn = '_num\d+'
    stride_ptn = '_stride\d+'
    jumps_ptn = '_jumps\d+'
    input_dim = re.findall('\d+', re.findall(inpd_ptn, model_name)[0])[0]
    input_length = re.findall('\d+', re.findall(inpl_ptn, model_name)[0])[0]
    num_steps = re.findall('\d+', re.findall(step_ptn, model_name)[0])[0]
    stride = re.findall('\d+', re.findall(stride_ptn, model_name)[0])[0]
    jumps = re.findall('\d+', re.findall(jumps_ptn, model_name)[0])[0]
    lr = re.findall('\d+\.\d+', re.findall('lr\d+\.\d+', model_name)[0])[0]
    return {"input_dim": int(input_dim), "input_length": int(input_length), "lr": float(lr),
            "num_steps": int(num_steps), "stride": int(stride), "jumps": int(jumps)}


model_name = "Koopman_megatrawl_seed901_jumps64_poly3_sin5_" \
             "exp2_lr0.0001_decay0.95_dim8_inp64_pred16_num32_enchid1024_" \
             "dechid1024_trm1024_conhid1024_enclys3_declys3_trmlys3_conlys3_latdim64_" \
             "rm=5_cm=10_RevINTrue_insnormFalse_globalKFalse_contKFalse_stride8.pt"
print(parse_model_name(model_name))
