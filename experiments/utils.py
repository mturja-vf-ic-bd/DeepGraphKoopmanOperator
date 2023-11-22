import re
import unittest


def parse_model_name(model_name):
    inpd_ptn = '_dim\d+'
    inpl_ptn = '_inp\d+'
    step_ptn = '_num\d+'
    jumps_ptn = '_jumps\d+'
    seed_ptn = '_seed\d+'
    try:
        input_dim = re.findall('\d+', re.findall(inpd_ptn, model_name)[0])[0]
        input_length = re.findall('\d+', re.findall(inpl_ptn, model_name)[0])[0]
        num_steps = re.findall('\d+', re.findall(step_ptn, model_name)[0])[0]
        jumps = re.findall('\d+', re.findall(jumps_ptn, model_name)[0])[0]
        lr = re.findall('\d+\.\d+', re.findall('lr\d+\.\d+', model_name)[0])[0]
        decay = re.findall('\d+\.\d+', re.findall('decay\d+\.\d+', model_name)[0])[0]
        # rm = re.findall('\d+', re.findall('_rm=\d+', model_name)[0])[0]
        seed = re.findall('\d+', re.findall(seed_ptn, model_name)[0])[0]
        return {"input_dim": int(input_dim), "input_length": int(input_length), "lr": float(lr),
                "num_steps": int(num_steps), "jumps": int(jumps), "decay":
                    float(decay), "rm": 0, "seed": int(seed)}
    except:
        print("An exception occured. Probably no match")
    return None


class test_parse_model_name(unittest.TestCase):
    def test_print_outputs(self):
        # model_name = "Koopman_megatrawl_seed901_jumps64_" \
        #              "poly3_sin5_exp2_lr0.0003_decay1.0_dim32_" \
        #              "inp512_pred32_num32_enchid1024_dechid1024_trm1024_" \
        #              "conhid1024_enclys3_declys3_trmlys3_conlys3_latdim64_rm=10_" \
        #              "cm=20_RevINTrue_insnormFalse_globalKFalse_contKFalse_stride32.pt"
        model_name = "SpKNF_megatrawl_seed34_jumps128_poly2_sin10_exp2_lr0.0003_decay1.0_dim8_inp128_pred32_num32_latdim64_RevINTrue_insnormFalse_globalKTrue_stride4_rank2048_sbsc128.pt"
        print(parse_model_name(model_name))
