import os
import sys
import numpy as np
import torch
import triton_python_backend_utils as pb_utils

from rdkit import Chem
from torch_geometric.data import Data


#sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "openpom_src"))
#from src.models.gcn_model import GCNModel


def smiles_to_pyg_graph(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")

    num_atoms = mol.GetNumAtoms()
    atom_features = []

    for atom in mol.GetAtoms():
        atom_features.append([atom.GetAtomicNum()])

    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        edge_index.append([start, end])
        edge_index.append([end, start])
        edge_attr.append([bond.GetBondTypeAsDouble()])
        edge_attr.append([bond.GetBondTypeAsDouble()])

    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


class TritonPythonModel:
    def initialize(self, args):
        model_path = os.path.join(os.path.dirname(__file__), "example_model.pt")
        self.model = torch.load(model_path, map_location=torch.device("cpu"))
#        self.model.eval()

    def execute(self, requests):
        responses = []

        for request in requests:
            try:
                smiles_input = pb_utils.get_input_tensor_by_name(request, "SMILES").as_numpy()[0]
                smiles_str = smiles_input.decode("utf-8")

                graph_data = smiles_to_pyg_graph(smiles_str)
                graph_data = graph_data.to("cpu")
                graph_data.batch = torch.zeros(graph_data.num_nodes, dtype=torch.long)

                with torch.no_grad():
                    output = self.model(graph_data)
                    result = output.squeeze().cpu().numpy()

                output_tensor = pb_utils.Tensor("OUTPUT", result.astype(np.float32))
                inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])

            except Exception as e:
                error_response = pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(str(e))
                )
                responses.append(error_response)
                continue

            responses.append(inference_response)

        return responses
