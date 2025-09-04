import torch
from typing import *
from jaxtyping import Float as FT32
import torchvision
from rich import print as rprint # 增强的print

Tensor = torch.Tensor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


csv_file_dict = {
        "CC": "/root/autodl-tmp/project_gnn_original_data/formal_verification_data/CC/csv/cc_gnn_new.csv",
        "FPV": "/root/autodl-tmp/project_gnn_original_data/formal_verification_data/FPV/csv/fpv_gnn.csv",
        "UNR": "/root/autodl-tmp/project_gnn_original_data/formal_verification_data/UNR/csv/unr_gnn_new.csv",
        "BV": "/root/autodl-tmp/project_gnn_original_data/smt_lib/BV/csv/new_bv.csv",
        "NRA": "/root/autodl-tmp/project_gnn_original_data/smt_lib/NRA/csv/new_nra.csv",
        "QF_BVFPLRA": "/root/autodl-tmp/project_gnn_original_data/smt_lib/QF_BVFPLRA/csv/new_qf_bvfplra.csv",
        "QF_UFBV": "/root/autodl-tmp/project_gnn_original_data/smt_lib/QF_UFBV/csv/new_qf_ufbv.csv",
        "QF_LIA": "/root/autodl-tmp/project_gnn_original_data/smt_lib/QF_LIA/csv/new_qf_lia.csv"
    }

save_path_dict = {
        "CC": "/root/autodl-tmp/project_gnn_original_data/formal_verification_data/CC/bin", 
        "FPV": "/root/autodl-tmp/project_gnn_original_data/formal_verification_data/FPV/bin",
        "UNR": "/root/autodl-tmp/project_gnn_original_data/formal_verification_data/UNR/bin",
        "BV": "/root/autodl-tmp/project_gnn_original_data/smt_lib/BV/bin",
        "NRA": "/root/autodl-tmp/project_gnn_original_data/smt_lib/NRA/bin",
        "QF_BVFPLRA": "/root/autodl-tmp/project_gnn_original_data/smt_lib/QF_BVFPLRA/bin",
        "QF_UFBV": "/root/autodl-tmp/project_gnn_original_data/smt_lib/QF_UFBV/bin",
        "QF_LIA": "/root/autodl-tmp/project_gnn_original_data/smt_lib/QF_LIA/bin"
    }


solver_names = {
        
        "CC": {
            "z3",
            "yices-2.6.2",
            "yices-2.6.5",
            "cvc4",
            "cvc5",
            "mathsat"
        },
        "FPV": {
            "z3",
            "yices_2_6_5",
            "yices_2_6_2",
            "cvc4",
            "cvc5",
            "mathsat"
        },
        "UNR": {
            "z3",
            "yices-2.6.2",
            "yices-2.6.5",
            "cvc4",
            "cvc5",
            "mathsat"
        },
        "QF_LIA": {
            "veriT",
            "smtinterpol-2.5-671-g6d0a7c6e",
            "CVC4-sq-final",
            "z3-4.8.8",
            "Yices 2.6.2 for SMTCOMP2020",
            "MathSAT5"
        },
        "BV": {
            "Boolector-wrapped-sq",
            "CVC4-2019-06-03-d350fe1-wrapped-sq",
            "Poolector-wrapped-sq",
            "Q3B-wrapped-sq",
            "UltimateEliminator+MathSAT-5.5.4-wrapped-sq",
            "z3-4.8.4-d6df51951f4c-wrapped-sq"
        },
        "NRA": {
            "CVC4-2019-06-03-d350fe1-wrapped-sq",
            "UltimateEliminator+MathSAT-5.5.4-wrapped-sq",
            "UltimateEliminator+Yices-2.6.1-wrapped-sq",
            "vampire-4.4-smtcomp-wrapped-sq",
            "z3-4.8.4-d6df51951f4c-wrapped-sq"
        },
        "QF_BVFPLRA": {
            "COLIBRI 20.5.25",
            "CVC4-sq-final",
            "MathSAT5"
        },
        "QF_UFBV": {
            "CVC4-sq-final",
            "z3-4.8.8",
            "Bitwuzla",
            "Yices 2.6.2 for SMTCOMP2020",
            "MathSAT5"
        }
    }

if __name__ == "__main__":
    rprint("[bold red]这是加粗的红色文本[/bold red]")
    rprint("[blue]这是蓝色文本[/blue]")