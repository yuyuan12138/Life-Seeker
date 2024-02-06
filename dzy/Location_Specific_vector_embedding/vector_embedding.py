from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np


def generate_morgan_fingerprint(molecule):
    mol = Chem.MolFromSmiles(molecule)

    if mol is not None:
        # 生成分子的Morgan Circular Fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)  # 半径为2的Morgan Fingerprint
        arr = np.zeros((1,))
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)

        return arr
    else:
        return None

molecule_smiles = "OC1C(N2C=3C(N=C2)=C(N)N=CN3)OC(C[S+](CCC(C([O-])=O)N)C)C1O"
vector_representation = generate_morgan_fingerprint(molecule_smiles)

print("分子向量表示:", vector_representation.shape)

