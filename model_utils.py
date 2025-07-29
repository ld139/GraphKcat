import os
import pickle
import torch
import numpy as np
import pandas as pd
from rdkit import rdBase
import torch
from io import StringIO
import sys
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.PDBIO import Select

def read_mol(sdf_fileName, mol2_fileName, verbose=False):
    rdBase.LogToPythonStderr()
    stderr = sys.stderr
    sio = sys.stderr = StringIO()
    mol = Chem.MolFromMolFile(sdf_fileName, sanitize=False)
    problem = False
    try:
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
        sm = Chem.MolToSmiles(mol)
    except Exception as e:
        sm = str(e)
        problem = True
    if problem:
        mol = Chem.MolFromMol2File(mol2_fileName, sanitize=False)
        problem = False
        try:
            Chem.SanitizeMol(mol)
            mol = Chem.RemoveHs(mol)
            sm = Chem.MolToSmiles(mol)
            problem = False
        except Exception as e:
            sm = str(e)
            problem = True

    if verbose:
        print(sio.getvalue())
    sys.stderr = stderr
    return mol, problem
# vocab_dict = {'O=C(NC1=CC=CC=C1)C1=CC=CC=C1': 0, 'C1=CC=C(CCCC2=CC=CC=C2)C=C1': 1, 'O=C(C1=CC=CC=C1)N1CCNCC1': 2, 'CCCCC(=O)NC1=CC=CC=C1': 3, 'C1=CC=C(CC2=CC=CC=C2)C=C1': 4, 'CCCNC(=O)C1=CC=CC=C1': 5, 'CCCC(=O)NC1=CC=CC=C1': 6, 'CCCCC1=CC=C(OC)C=C1': 7, 'C1=CC=C(C2=CC=CC=C2)C=C1': 8, 'CCCCCCC1=CC=CC=C1': 9, 'O=C1NC(S)=NC2=CC=CC=C12': 10, 'CC(NC(N)=O)C1=CC=CC=C1': 11, 'CCCCCC1=CC=CC=C1': 12, 'CCNC(=O)C1=CC=CC=C1': 13, 'CCC(=O)NC1=CC=CC=C1': 14, 'CC1=CC=CC=C1CCC=O': 15, 'NC(=O)CCC1=CC=CC=C1': 16, 'NC(=O)NCC1=CC=CC=C1': 17, 'O=CCCCC1=CC=CC=C1': 18, 'CCC1=CNC2=CC=CC=C12': 19, 'O=CC1CC1C1=CC=CC=C1': 20, 'COC1=CC=C(C)C=C1OC': 21, 'CCCC1=CC=C(OC)C=C1': 22, 'CC1=CC=CC2=CC=CC=C12': 23, 'O=C1NC=NC2=CC=CC=C12': 24, 'CNC(=O)CC1=CC=CC=C1': 25, 'CCCCC1=CC=CC=C1': 26, 'O=CCCC1=CC=CC=C1': 27, 'CNC(=O)C1=CC=CC=C1': 28, 'NC(=O)CC1=CC=CC=C1': 29, 'FC(F)(F)C1=CC=CC=C1': 30, 'CC(=O)NC1=CC=CC=C1': 31, 'C1=CC=C2CCCCC2=C1': 32, 'CC1CC1C1=CC=CC=C1': 33, 'CC1=CC=C(C(N)=O)C=C1': 34, 'C1=CC=C2C=CC=CC2=C1': 35, 'NC(=O)NC1=CC=CC=C1': 36, 'NC1=NC2=CC=CC=C2S1': 37, 'CCC1=CC=C(OC)C=C1': 38, 'O=[SH](=O)NC1=CC=CC=C1': 39, 'CC1=CC=CC=C1C(N)=O': 40, 'NNC(=O)C1=CC=CC=C1': 41, 'C1=CN=C2C=CC=CC2=C1': 42, 'COC(=O)C1=CC=CC=C1': 43, 'CC1=CC=C2OCOC2=C1': 44, 'CC1=CC=CC(C(F)F)=C1': 45, 'O=CC1=CC=CC=C1C=O': 46, 'COC1=CC=C(CN)C=C1': 47, 'O=CC=CC1=CC=CC=C1': 48, 'CCC(=O)N1CCNCC1': 49, 'CCCC1=CC=CC=C1C': 50, 'CC1=CC=CC(C(N)=O)=C1': 51, 'NS(=O)(=O)C1=CC=CC=C1': 52, 'CC1=CC(C(N)=O)=CC=C1': 53, 'NC(=O)C1=CC=CC=C1Cl': 54, 'CC1CC2CCCC(C1)C2': 55, 'FC(F)OC1=CC=CC=C1': 56, 'NC(=O)C1=CC=CC=C1F': 57, 'CCC1=CC=C(CC)C=C1': 58, 'CCC1=CC(OC)=CC=C1': 59, 'NC(=NO)C1=CC=CC=C1': 60, 'CC1=CNC2=CC=CC=C12': 61, 'CCC(=O)C1=CC=CC=C1': 62, 'COC1=CC=C(C=O)C=C1': 63, 'CC1=CC=CC=C1C(F)F': 64, 'CCCC1=CC=CC=C1': 65, 'NC(=O)C1=CC=CC=C1': 66, 'O=CCC1=CC=CC=C1': 67, 'CCC1=CC=CC=C1C': 68, 'FC(F)C1=CC=CC=C1': 69, 'CC1=CC=CC=C1C=O': 70, 'COC1=CC=C(C)C=C1': 71, 'C1=CC=C2CCCC2=C1': 72, 'CC(=O)C1=CC=CC=C1': 73, 'CC=CC1=CC=CC=C1': 74, 'CC1=CC=C(C=O)C=C1': 75, 'NCCC1=CC=CC=C1': 76, 'CC(C)C1=CC=CC=C1': 77, 'O=C(O)C1=CC=CC=C1': 78, 'CCC1=CC=C(F)C=C1': 79, 'COC1=CC=CC(C)=C1': 80, 'NC(=O)C1=CC=CN=C1': 81, 'NN=CC1=CC=CC=C1': 82, 'CC1=CC=CC(CF)=C1': 83, 'CN(N)C1=CC=CC=C1': 84, 'COC1=CC=CC=C1C': 85, 'CCOC1=CC=CC=C1': 86, 'FCOC1=CC=CC=C1': 87, 'CCC1=CC=C(Cl)C=C1': 88, 'CCC1=CC=C(C)C=C1': 89, 'CC1=CC=C(CN)C=C1': 90, 'CCC1=CC(C)=CC=C1': 91, 'CCCCC(=O)NCC': 92, 'NC(=O)C1=CC=NC=C1': 93, 'O=[SH](=O)C1=CC=CC=C1': 94, 'NC(=O)C1CNC(=O)C1': 95, 'CC(O)C1=CC=CC=C1': 96, 'CC(=O)N1CCNCC1': 97, 'O=[SH](=O)N1CCNCC1': 98, 'CCCNC(=O)CCC': 99, 'CNCC1=CC=CC=C1': 100, 'O=[SH](=O)N1CCCCC1': 101, 'NCC1=CC=C(F)C=C1': 102, 'CC1CCCCC1C=O': 103, 'CC1=CC=CC=C1CF': 104, 'CCC1=CC=CC(C)=C1': 105, 'NC1=CC=CC=C1C=O': 106, 'CCC1=CC=CC=C1F': 107, 'CC1=CC=C(C)C(C)=C1': 108, 'COC1=CC=C(N)C=C1': 109, 'COC1=CC(C)=CC=C1': 110, 'NC(=O)N1CCNCC1': 111, 'CCNC1=NC=CC=N1': 112, 'CC1=CC(C=O)=CC=C1': 113, 'CC1=CC=C(CF)C=C1': 114, 'C1=CC=C(C2CC2)C=C1': 115, 'CCC1=C(C)ON=C1C': 116, 'O=CC1=CC=CC=C1Cl': 117, 'O=[SH](=O)N1CCOCC1': 118, 'C1=CC=C2NC=CC2=C1': 119, 'O=CCC1=CCCCC1': 120, 'CCC1=NN(C)C=C1C': 121, 'CC1=CC=C(N)C(C)=C1': 122, 'CC=CC=CC=CC=O': 123, 'C1=CC=C2NC=NC2=C1': 124, 'CC=C(C)C(=O)NCC': 125, 'CCC1=CC=CC=C1': 135, 'O=CC1=CC=CC=C1': 127, 'CC1=CC=CC=C1C': 128, 'NCC1=CC=CC=C1': 129, 'CC1=CC=C(C)C=C1': 130, 'COC1=CC=CC=C1': 131, 'CC1=CC=CC(C)=C1': 175, 'FCC1=CC=CC=C1': 133, 'OCC1=CC=CC=C1': 134, 'CC1=CC=C(F)C=C1': 136, 'CCCNC(=O)CC': 137, 'CC1=CC=CC=C1Cl': 138, 'CC1=CC=C(Cl)C=C1': 139, 'N#CC1=CC=CC=C1': 140, 'CC1=CC=CC=C1F': 141, 'CC1=CC=CC=C1O': 142, 'NCC1=CC=CN=C1': 143, 'FC1=CC=CC(F)=C1': 144, 'NC1=CC=C(F)C=C1': 145, 'CCCC(=O)NCC': 146, 'CCC1=CC=CC=N1': 147, 'CCCCNC(N)=O': 148, 'NC1=CC=CC=C1S': 149, 'CC1CNCC(C)O1': 150, 'NCC1=CC=NC=C1': 151, 'CC1=CC=CC(F)=C1': 152, 'CC1=CC=C(O)C=C1': 153, 'O=CC1CCCCC1': 154, 'CCCCCCC=O': 155, 'NNC1=CC=CC=C1': 156, 'CC1=CC=CC(Cl)=C1': 157, 'CNC1=CC=CC=C1': 158, 'CC1=CC=CC=C1N': 159, 'CCC1=CC=NC=C1': 160, 'CC(C)(C)OC(N)=O': 161, 'ClC1=CC=CC(Cl)=C1': 162, 'CCNC(=O)C(N)=O': 163, 'O=CC1CNC(=O)C1': 164, 'CCC1=CC=CN=C1': 165, 'O=CC1=CC=CN=C1': 166, 'CC1=CC=C(C=O)O1': 167, 'O=CC1=CC=CC=N1': 168, 'O=CC1=CC=NC=C1': 169, 'CCCCC(=O)NC': 170, 'CCCC(=O)OCC': 171, 'CC=C(C)C(F)(F)F': 172, 'CCCCCC(N)=O': 173, 'CC(=O)NCC(N)=O': 174, 'NC(=O)CNC(N)=O': 176, 'NC1=CC=C(Cl)C=C1': 177, 'CC1=CC=C(N)C=C1': 178, 'NC(=O)C1=CC=CO1': 179, 'NC(=O)C1=CC=CS1': 180, 'CCN1CCNCC1': 181, 'CC1CNCC(C)C1': 182, 'CC1=CC(F)=CC=C1': 183, 'O=[SH](=O)N1CCCC1': 184, 'FC1=CC=CC=C1F': 185, 'CC=CCC(F)(F)F': 186, 'FCC1CCCCC1': 187, 'CC1=CC(Cl)=CC=C1': 188, 'CC(C)CNC(N)=O': 189, 'NC1=CC=CC=C1F': 190, 'NC(=O)CCCC=O': 191, 'CC1=CC=CC=C1': 192, 'NC1=CC=CC=C1': 193, 'FC1=CC=CC=C1': 194, 'ClC1=CC=CC=C1': 195, 'CC1=CC=CN=C1': 196, 'OC1=CC=CC=C1': 197, 'CCCCCC=O': 198, 'CCNC(=O)CC': 199, 'CC=CC=CC=O': 200, 'CCCCC(N)=O': 201, 'CC1=CC=NC=C1': 202, 'CCCNC(N)=O': 203, 'CC1=CC=CC=N1': 204, 'CCNCC(N)=O': 205, 'O=CC1=CC=CS1': 206, 'CC1CCCNC1': 207, 'C=CC(=CC)OC': 208, 'CC=C(C)C(N)=O': 209, 'CC=C(C)C(F)F': 210, 'BrC1=CC=CC=C1': 211, 'CCNC(=O)CN': 212, 'NC(=O)CCC=O': 213, 'NC1=CC=CC=N1': 214, 'O=CC1=CC=CO1': 215, 'CCCC(=O)NC': 216, 'NC1=NC=CC=N1': 217, 'CCCC(=O)OC': 218, 'CC1CNCCO1': 219, 'O=S1(=O)CCCC1': 220, 'CC1=CC(C)=NN1': 221, 'SC1=CC=CC=C1': 222, 'CCOC(=O)CC': 223, 'CCC1=CC=CS1': 224, 'CC=CCC(F)F': 225, 'CC=CC=C(C)C': 226, 'OC1CCCCC1': 227, 'CC1CCOC1C': 228, 'O=CC1CCCC1': 229, 'CC(=O)NCCN': 230, 'NC(=O)NCCO': 231, 'CNCC(=O)NC': 232, 'CC1=CC(C)=NO1': 233, 'NCSCC(N)=O': 234, 'CCCCC(=O)O': 235, 'CC(C)CC(N)=O': 236, 'CC1CCCCC1': 237, 'CCCN[SH](=O)=O': 238, 'CC1CCNCC1': 239, 'NC1=NN=C(S)S1': 240, 'O=C1CSC(=S)N1': 241, 'CCC1=CC=CO1': 242, 'CC(C)NC(N)=O': 243, 'NCC1=CC=CS1': 244, 'O=CCCCC=O': 245, 'SC1=CC=CC=N1': 246, 'CC(=O)NC(C)C': 247, 'CC1=CC=C(C)O1': 248, 'CCNC(=O)CO': 249, 'CC1=NNC(N)=C1': 250, 'CC1=C(C)SC=C1': 251, 'O=C1CCC(=O)N1': 252, 'CCCCC(C)C': 253, 'CCCCC=O': 254, 'C1=CC=NC=C1': 255, 'CCCC(N)=O': 256, 'CCNC(N)=O': 257, 'CCNC(C)=O': 258, 'CC=C(C)C=O': 259, 'CCC(=O)NC': 260, 'CC1=CC=CS1': 261, 'C1COCCN1': 262, 'CC1CCCO1': 263, 'CCC=C(C)C': 264, 'CC1=CC=CO1': 265, 'CC=C(C)CC': 266, 'NC(=O)C(N)=O': 267, 'CCN[SH](=O)=O': 268, 'CC=CC(N)=O': 269, 'CC=C(C)CF': 270, 'O=CCCC=O': 271, 'O=C1CSCN1': 272, 'NCCC(N)=O': 273, 'CCCC(C)C': 274, 'NC(=O)NC=O': 275, 'CC(N)=CC=O': 276, 'CCCC(=O)O': 277, 'CNCC(N)=O': 278, 'CC=CCCF': 279, 'CCC(=O)OC': 280, 'CC(C)C(N)=O': 281, 'C1=CCCC=C1': 282, 'NC1=CC=NN1': 283, 'C1CCNCC1': 284, 'O=C1CCCN1': 285, 'CCCCCO': 286, 'NNCC(N)=O': 287, 'CC=C(C)CN': 288, 'NC1=NC=NN1': 289, 'CC(O)C(N)=O': 290, 'CC=C(C)OC': 291, 'CNC(=O)CN': 292, 'CCCNCC': 293, 'CC1CC1C=O': 294, 'CC=CC=NN': 295, 'CCNCC=O': 296, 'CCC=C(C)N': 297, 'CNC(=O)NC': 298, 'CCCC(C)O': 299, 'CN(C)[SH](=O)=O': 300, 'CNC(=O)CS': 301, 'CCC(F)(F)F': 302, 'CC1=NN=NN1': 303, 'N#CCC(N)=O': 304, 'CC=C(N)C=O': 305, 'NC1=NC=CS1': 306, 'CC(N)C(N)=O': 307, 'CCCCCN': 308, 'CC=C(C)C#N': 309, 'CSCC(N)=O': 310, 'CC1=CC=NO1': 311, 'CCOC(C)=O': 312, 'NC(=O)CC=O': 313, 'OCC(F)(F)F': 314, 'CCC(O)CC': 315, 'NC1=NN=NN1': 316, 'CC(S)C(N)=O': 317, 'CCC(C)(C)C': 318, 'CC=C(N)CF': 319, 'CCCC(C)=O': 320, 'CCC(O)CF': 321, 'CCCCOC': 322, 'SC1=NN=CN1': 323, 'CN1C=CN=C1': 324, 'CC(=O)NCS': 325, 'NCC(F)(F)F': 326, 'CC(C)(C)CN': 327, 'CCC(N)=NO': 328, 'CC(C)N(C)N': 329, 'O=C1NCCO1': 330, 'CCCC=O': 331, 'CC=C(C)C': 332, 'CCC(N)=O': 333, 'CC=CC=O': 334, 'NCC(N)=O': 335, 'CNC(N)=O': 336, 'CC=CCC': 337, 'CNC(C)=O': 338, 'C=CC=CC': 339, 'NC(=O)CO': 340, 'NC(=O)CS': 341, 'CCC(C)C': 342, 'CC=C(C)N': 343, 'C1=CNN=C1': 344, 'C1=CSC=C1': 345, 'CC(C)CN': 346, 'CCC(C)O': 347, 'CCNCN': 348, 'CCC(=O)O': 349, 'NN=C(N)S': 350, 'C1=NN=NN1': 351, 'CCN(C)N': 352, 'CN[SH](=O)=O': 353, 'CCCCO': 354, 'C1=NC=NN1': 355, 'NCCC=O': 356, 'CC(C)(C)O': 357, 'CCCCN': 358, 'CCNC=O': 359, 'CCCNC': 360, 'CC(C)CO': 361, 'OCC(F)F': 362, 'CC(N)=CS': 363, 'C1=CNC=N1': 364, 'CCC(F)F': 365, 'CCNCS': 366, 'CN(N)CN': 367, 'CCCON': 368, 'NCC1CC1': 369, 'CCC(C)=O': 370, 'NCC(F)F': 371, 'CCCOC': 372, 'CC[SH](=O)=O': 373, 'CCSCN': 374, 'NC=NCN': 375, 'CS(N)(=O)=O': 376, 'CNC=CN': 377, 'C1=COC=C1': 378, 'CC(=O)NN': 379, 'CC(O)CN': 380, 'CC=CC=N': 381, 'O=CC1CC1': 382, 'CCCNN': 383, 'CC(N)=NO': 384, 'NC(=O)C=O': 385, 'C1=CNN=N1': 386, 'CNCCF': 387, 'CCCSC': 388, 'CNCCO': 389, 'CC(O)CO': 390, 'NCC(=O)O': 391, 'CNC=NN': 392, 'CCCC#N': 393, 'C1CCNC1': 394, 'CNC(N)=S': 395, 'C1=NN=CN1': 396, 'COCCO': 397, 'NNC(N)=S': 398, 'NNCCF': 399, 'CN=C(N)N': 400, 'CC=C(C)O': 401, 'C1CCOC1': 402, 'C1=CNC=C1': 403, 'CNC(C)C': 404, 'NNCC=O': 405, 'CC(C)(C)C': 406, 'CC=CC': 407, 'CCCN': 408, 'CC(N)=O': 409, 'NC(N)=O': 444, 'CCNC': 411, 'CCC=O': 412, 'CCCO': 413, 'N[SH](=O)=O': 414, 'CC(C)O': 415, 'CC=CN': 416, 'C[SH](=O)=O': 417, 'CC(C)C': 455, 'NCNN': 419, 'NCC=O': 420, 'CCNN': 421, 'CC=NC': 422, 'CN(C)N': 423, 'CCON': 424, 'CNC=O': 425, 'CC(=O)O': 426, 'CSCN': 427, 'OCCF': 428, 'CCC#N': 429, 'CCCF': 430, 'NCCO': 431, 'O=CCO': 432, 'CCCS': 433, 'CC=CS': 434, 'OC(F)F': 435, 'CNCS': 436, 'NC(N)=S': 437, 'NCCF': 438, 'N=C(N)S': 439, 'NC=CS': 440, 'CN=CN': 441, 'NNCS': 442, 'CCCC': 443, 'CCSC': 445, 'CC(C)=N': 446, 'CC=CO': 447, 'NN=CO': 448, 'NCCS': 449, 'CC(C)N': 450, 'NC=NN': 451, 'CC[SH]=O': 452, 'CNCN': 453, 'CCOC': 454, 'FCCS': 456, 'NC=CN': 457, 'O=CCS': 458, 'CCN': 459, 'CCO': 460, 'O=S=O': 461, 'CNN': 462, 'NCS': 463, 'NC=O': 464, 'CCC': 465, 'CCS': 466, 'CNC': 467, 'OCF': 468, 'CC=O': 469, 'N=CN': 470, 'O=CO': 471, 'NCO': 472, 'CC=N': 473, 'C=CC': 474, 'FCF': 475, 'C[SH]=O': 476, 'CON': 477, 'CCF': 478, 'NCN': 479, 'CC#N': 480, 'CC': 481, 'CN': 482, 'CO': 483, 'O=S': 484, 'CS': 485, 'NN': 486, 'NO': 487, 'CF': 488, 'NS': 489, 'B': 490, 'Br': 491, 'C': 492, 'Cl': 493, 'F': 494, 'I': 495, 'N': 496, 'O': 497, 'P': 498, 'S': 499}
vocab_dict = {'O=C(Nc1ccccc1)c1ccccc1': 0, 'O=C(c1ccccc1)N1CCNCC1': 1, 'CCN(C)C(=O)c1ccccc1': 2, 'CCCNC(=O)c1ccccc1': 3, 'O=CCNC(=O)c1ccccc1': 4, 'CCNC(=O)Nc1ccccc1': 5, 'CCNC(=O)c1ccccc1': 6, 'COc1ccc(C)cc1OC': 7, 'O=c1ncnc2ccccc12': 8, 'c1ccc(-n2cccn2)cc1': 9, 'CCC(=O)Nc1ccccc1': 10, 'CC(=O)NCc1ccccc1': 11, 'CCNC(=O)c1cccnc1': 12, 'CC(=O)Nc1ccccc1': 13, 'FC(F)(F)c1ccccc1': 14, 'c1ccc2ncccc2c1': 15, 'c1ccc2ccccc2c1': 16, 'CNC(=O)c1ccccc1': 17, 'COc1ccccc1OC': 18, 'c1ccc2occcc2c1': 19, 'c1ccc2c(c1)OCCO2': 20, 'FC(F)Oc1ccccc1': 21, 'COC(=O)c1ccccc1': 22, 'COc1cccc(OC)c1': 23, 'Cc1ccc2c(c1)OCO2': 24, 'CCC(=O)N1CCNCC1': 25, 'c1ccc2nccnc2c1': 26, 'Cc1cccc([N+](=O)[O-])c1': 27, 'NC(=O)c1ccccc1': 28, 'FC(F)c1ccccc1': 29, 'COc1cccc(C)c1': 30, 'c1ccc2[nH]ccc2c1': 31, 'c1ccc2scnc2c1': 32, 'CCOc1ccccc1': 33, 'c1ccc2ncnc2c1': 34, 'COc1ccc(C)cc1': 35, 'O=[N+]([O-])c1ccccc1': 36, 'FCOc1ccccc1': 37, 'O=CNc1ccccc1': 38, 'c1ccc2occc2c1': 39, 'COc1ccccc1C': 40, 'CCCNC(=O)NCC': 41, 'c1ccc2[nH]cnc2c1': 42, 'O=C([O-])c1ccccc1': 43, 'c1ccc2nccc2c1': 44, 'c1ccc2ocnc2c1': 45, 'CC(=O)N1CCCCC1': 46, 'c1ccc2c(c1)OCO2': 47, 'CC(=O)N1CCNCC1': 48, 'CCCc1ccccc1': 49, 'CC(=O)N1CCOCC1': 50, 'O=[SH](=O)N1CCNCC1': 51, 'CCNC(=O)N(C)CC': 52, 'NC(=O)C1CCNCC1': 53, 'Cc1cc(C)cc(C)c1': 54, 'O=Cc1ccccc1': 55, 'COc1ccccc1': 56, 'FCc1ccccc1': 57, 'CCNC(=O)NCC': 58, 'Cc1cccc(C)c1': 59, 'CCc1ccccc1': 60, 'Cc1ccccc1F': 61, 'Cc1ccccc1C': 62, 'Cc1cccc(F)c1': 63, 'Cc1ccccc1Cl': 64, 'CCN1CCOCC1': 65, 'Fc1cccc(F)c1': 66, 'N#Cc1ccccc1': 67, 'Cc1ccc(F)cc1': 68, 'Cc1ccc(C)cc1': 69, 'Cc1cccc(Cl)c1': 70, 'CCCC(=O)NCC': 71, 'ccc1ccccc1': 72, 'Cc1ccc(Cl)cc1': 73, 'Cc1ccccc1O': 74, 'O=CN1CCNCC1': 75, 'Fc1ccccc1F': 76, 'OCc1ccccc1': 77, 'Clc1cccc(Cl)c1': 78, 'O=CN1CCCCC1': 79, 'NCc1ccccc1': 80, 'Cc1cccc(O)c1': 81, 'CCCCNC(C)=O': 82, 'CCC(=O)N(C)CC': 83, 'Clc1ccccc1Cl': 84, 'Cc1ccc(O)cc1': 85, 'CCNC(=O)C(N)=O': 86, 'CC(=O)N1CCCC1': 87, 'CCc1ccn(C)n1': 88, 'CCN1CSCC1=O': 89, 'CCNC(=O)C(C)C': 90, 'CCN1CCNCC1': 91, 'NC(=O)c1ccco1': 92, 'Cc1ccccc1': 93, 'Fc1ccccc1': 94, 'Clc1ccccc1': 95, 'Oc1ccccc1': 96, 'CCNC(=O)CC': 97, 'Nc1ccccc1': 98, 'Brc1ccccc1': 99, 'Cc1cc(C)nn1': 100, 'Cc1ccccn1': 101, 'Cc1ccncn1': 102, 'CCCNC(C)=O': 103, 'O=S1(=O)CCCC1': 104, 'CCNC(=O)NC': 105, 'Cc1cccnc1': 106, 'CCN(C)C(C)=O': 107, 'O=c1ccnc[nH]1': 108, 'C[C@@H]1CCCCC1': 109, 'Clc1cccnc1': 110, 'CC1CCCCC1': 111, 'Cc1cc(C)on1': 112, 'O=c1ccccn1': 113, 'Sc1ccccc1': 114, 'CCOC(=O)CC': 115, 'C[NH+]1CCNCC1': 116, 'CCCNC(N)=O': 117, 'C[NH+]1CCCCC1': 118, 'O=c1ccncn1': 119, 'CC(=O)NC(C)C': 120, 'Cc1ccn(C)n1': 121, 'CN1CCNCC1': 122, 'CCN(C)C(N)=O': 123, 'CCNS(C)(=O)=O': 124, 'Cc1ccc(C)n1': 125, 'Clc1ccccn1': 126, 'CCNC(=O)CN': 127, 'O=c1cccnn1': 128, 'O=CN1CCCC1': 129, 'Cc1ccncc1': 130, 'Nc1ccncn1': 131, 'CCN(C)[SH](=O)=O': 132, 'CC(=O)NCC=O': 133, 'c1ccccc1': 134, 'c1ccncc1': 135, 'CCNC(C)=O': 136, 'c1cncnc1': 137, 'CCNC(N)=O': 138, 'Cc1ccnn1': 139, 'CCN[SH](=O)=O': 140, 'Cc1ccco1': 141, 'Cn1cccn1': 142, 'CCCC(N)=O': 143, 'CCCNC=O': 144, 'Cc1cscn1': 145, 'Cc1cccs1': 146, 'CCCC(C)C': 147, 'Cc1ccno1': 148, 'c1cnccn1': 149, 'c1cc[nH+]cc1': 150, 'C1COCCN1': 151, 'c1ccnnc1': 152, 'O=CCNC=O': 153, 'C1CCCCC1': 154, 'C1CCNCC1': 155, 'C1CNCCN1': 156, 'C1CC[NH+]CC1': 157, 'O=C1CCCN1': 158, 'C[C@@H]1CCCO1': 159, 'CCC(=O)NC': 160, 'CCN(C)C=O': 161, 'C[C@H]1CCCO1': 162, 'CCCNCC': 163, 'Cc1cccn1': 164, 'CCNCC=O': 165, 'Sc1ncnn1': 166, 'CCC(=O)OC': 167, 'C1C[NH+]CCN1': 168, 'Cc1ccsc1': 169, 'CCCC(=O)[O-]': 170, 'NC(=O)C(N)=O': 171, 'CC(C)NC=O': 172, 'O=C1CSCN1': 173, 'CC(=O)N(C)C': 174, 'Cc1nccs1': 175, 'Cc1ncnn1': 176, 'OC1CCCC1': 177, 'CCCCCC': 178, 'CC(C)C(N)=O': 179, 'CCC[NH+]CC': 180, 'Clc1cccs1': 181, 'CCC[NH2+]CC': 182, 'Cn1cnnc1': 183, 'Cc1ccc[nH]1': 184, 'ccccc=O': 185, 'CC(C)[NH+](C)C': 186, 'Cn1ccnc1': 187, 'Cn1cncn1': 188, 'CCNC=O': 189, 'c1cnnc1': 190, 'CCNCC': 191, 'c1cscn1': 192, 'c1ccsc1': 193, 'c1ncnn1': 194, 'ccccC': 195, 'CNC(C)=O': 196, 'CCC(N)=O': 197, 'ccccn': 198, 'c1cnoc1': 199, 'CCCC=O': 200, 'CCCCC': 201, 'c1ncon1': 202, 'c1cn[nH]c1': 203, 'c1ccoc1': 204, 'cccc[nH]': 205, 'C1CCCC1': 206, 'ccccc': 207, 'c1cncn1': 208, 'CCC(C)C': 209, 'c1nnnn1': 210, 'c1nncs1': 211, 'c1nnco1': 212, 'CCCCO': 213, 'NC(=O)CS': 214, 'cccco': 215, 'c1nc[nH]n1': 216, 'c1c[nH+]cn1': 217, 'NC(=O)C=O': 218, 'CCCNC': 219, 'c1cocn1': 220, 'NC(=O)CO': 221, 'c1cnnn1': 222, 'c1ccnc1': 223, 'CCCCN': 224, 'CCC(=O)[O-]': 225, 'CC[NH+]CC': 226, 'ncnc=O': 227, 'C1CCOC1': 228, 'C1CCNC1': 229, 'CC(C)(C)O': 230, 'CCOC=O': 231, 'C1CC[NH+]C1': 232, 'CCC[NH+]C': 233, 'CNCC=O': 234, 'CN[SH](=O)=O': 235, 'c1cc[nH]c1': 236, 'CNC(N)=O': 237, 'NCC(N)=O': 238, 'CC[NH+](C)C': 239, 'OCC(F)F': 240, 'CCOCC': 241, 'CCCC[NH2+]': 242, 'CC[SH](=O)=O': 243, 'CCC(C)=O': 244, 'c1c[nH]cn1': 245, 'COC(C)=O': 246, 'CCN(C)C': 247, 'CC(C)CO': 248, 'CC[C@H](C)O': 249, 'CCNCN': 250, 'CC[C@@H](C)O': 251, 'CC[NH2+]CC': 252, 'cccc': 253, 'CCNC': 254, 'CCCC': 255, 'CC(N)=O': 256, 'ccnc': 257, 'CCC=O': 258, 'cccn': 259, 'CC(C)C': 260, 'cncn': 261, 'CCCO': 262, 'CC[NH+]C': 263, 'CC(C)O': 264, 'C[SH](=O)=O': 265, 'N[SH](=O)=O': 266, 'CNC=O': 267, 'CCCN': 268, 'NCC=O': 269, 'CCC[NH2+]': 270, 'FC(F)F': 271, 'CCC[NH+]': 272, 'COC=O': 273, 'OCCF': 274, 'NC(N)=O': 275, 'CCOC': 276, 'O=CCO': 277, 'CC(=O)[O-]': 278, 'CC(=O)O': 279, 'CC(C)N': 280, 'nccn': 281, 'CCC#N': 282, 'ccc=O': 283, 'CC[NH2+]C': 284, '[N-][SH](=O)=O': 285, 'CCCF': 286, 'NNC=O': 287, 'cc(C)n': 288, 'CCCS': 289, 'Cncn': 290, 'NCCO': 291, 'CCN': 292, 'CCC': 293, 'ccn': 294, 'CCO': 295, 'O=S=O': 296, 'CC[NH+]': 297, 'NC=O': 298, 'ncn': 299, 'O=[N+][O-]': 300, 'CC=O': 301, 'ccc': 302, 'CC[NH2+]': 303, 'ncs': 304, 'CNC': 305, 'nco': 306, 'FCF': 307, 'C[NH+]C': 308, 'O=CO': 309, 'C1CC1': 310, 'CCS': 311, '[nH]c=O': 312, 'O=C[O-]': 313, 'nc[nH]': 314, 'CS=O': 315, 'OCF': 316, 'NC=S': 317, 'cnn': 318, 'CCF': 319, 'C=CC': 320, 'nc=O': 321, 'CC[NH3+]': 322, 'cc': 323, 'CC': 324, 'cn': 325, 'CO': 326, 'CN': 327, 'O=S': 328, 'C[NH+]': 329, '[N+][O-]': 330, 'CF': 331, 'CS': 332, 'C=O': 333, 'nn': 334, 'c[nH]': 335, 'C#N': 336, 'C[NH2+]': 337, 'NN': 338, 'B': 339, 'Br': 340, 'C': 341, 'Cl': 342, 'F': 343, 'I': 344, 'N': 345, 'O': 346, 'P': 347, 'S': 348, 'Na': 349}
def write_renumbered_sdf(toFile, sdf_fileName, mol2_fileName):
    # read in mol
    mol, _ = read_mol(sdf_fileName, mol2_fileName)
    # reorder the mol atom number as in smiles.
    m_order = list(mol.GetPropsAsDict(includePrivate=True, includeComputed=True)['_smilesAtomOutputOrder'])
    mol = Chem.RenumberAtoms(mol, m_order)
    w = Chem.SDWriter(toFile)
    w.write(mol)
    w.close()
def read_pdbbind_data(fileName):
    with open(fileName) as f:
        a = f.readlines()
    info = []
    for line in a:
        if line[0] == '#':
            continue
        lines, ligand = line.split('//')
        pdb, resolution, year, affinity, raw = lines.strip().split('  ')
        ligand = ligand.strip().split('(')[1].split(')')[0]
        # print(lines, ligand)
        info.append([pdb, resolution, year, affinity, raw, ligand])
    info = pd.DataFrame(info, columns=['pdb', 'resolution', 'year', 'affinity', 'raw', 'ligand'])
    info.year = info.year.astype(int)
    info.affinity = info.affinity.astype(float)
    return info
three_to_one = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 
                'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 
                'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}


def get_clean_res_list(res_list, verbose=False, ensure_ca_exist=False, bfactor_cutoff=None):
    res_list = [res for res in res_list if (('N' in res) and ('CA' in res) and ('C' in res) and ('O' in res))]
    clean_res_list = []
    for res in res_list:
        hetero, resid, insertion = res.full_id[-1]
        if hetero == ' ':
            if res.resname not in three_to_one:
                if verbose:
                    print(res, "has non-standard resname")
                continue
            if (not ensure_ca_exist) or ('CA' in res):
                if bfactor_cutoff is not None:
                    ca_bfactor = float(res['CA'].bfactor)
                    if ca_bfactor < bfactor_cutoff:
                        continue
                clean_res_list.append(res)
        else:
            if verbose:
                print(res, res.full_id, "is hetero")
    return clean_res_list

def remove_hetero_and_extract_ligand(res_list, verbose=False, ensure_ca_exist=False, bfactor_cutoff=None):
    # get all regular protein residues. and ligand.
    clean_res_list = []
    ligand_list = []
    for res in res_list:
        hetero, resid, insertion = res.full_id[-1]
        if hetero == ' ':
            if (not ensure_ca_exist) or ('CA' in res):
                # in rare case, CA is not exists.
                if bfactor_cutoff is not None:
                    ca_bfactor = float(res['CA'].bfactor)
                    if ca_bfactor < bfactor_cutoff:
                        continue
                clean_res_list.append(res)
        elif hetero == 'W':
            # is water, skipped.
            continue
        else:
            ligand_list.append(res)
            if verbose:
                print(res, res.full_id, "is hetero")
    return clean_res_list, ligand_list

def get_res_unique_id(residue):
    pdb, _, chain, (_, resid, insertion) = residue.full_id
    unique_id = f"{chain}_{resid}_{insertion}"
    return unique_id

def save_cleaned_protein(c, proteinFile):
    res_list = list(c.get_residues())
    clean_res_list, ligand_list = remove_hetero_and_extract_ligand(res_list)
    res_id_list = set([get_res_unique_id(residue) for residue in clean_res_list])

    io=PDBIO()
    class MySelect(Select):
        def accept_residue(self, residue, res_id_list=res_id_list):
            if get_res_unique_id(residue) in res_id_list:
                return True
            else:
                return False
    io.set_structure(c)
    io.save(proteinFile, MySelect())
    return clean_res_list, ligand_list

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def create_dir(dir_list):
    assert  isinstance(dir_list, list) == True
    for d in dir_list:
        if not os.path.exists(d):
            os.makedirs(d)

def save_model_dict(model, model_dir, msg):
    model_path = os.path.join(model_dir, msg + '.pt')
    torch.save(model.state_dict(), model_path)
    print("model has been saved to %s." % (model_path))

def load_model_dict(model, ckpt):
    model.load_state_dict(torch.load(ckpt))

def del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path,i)  
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)

def write_pickle(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def read_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj



class BestMeter(object):
    """Computes and stores the best value"""

    def __init__(self, best_type):
        self.best_type = best_type  
        self.count = 0      
        self.reset()

    def reset(self):
        if self.best_type == 'min':
            self.best = float('inf')
        else:
            self.best = -float('inf')

    def update(self, best):
        self.best = best
        self.count = 0

    def get_best(self):
        return self.best

    def counter(self):
        self.count += 1
        return self.count


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def get_average(self):
        self.avg = self.sum / (self.count + 1e-12)

        return self.avg
