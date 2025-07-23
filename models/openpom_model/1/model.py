import os
import sys
import numpy as np
import torch
import triton_python_backend_utils as pb_utils

from rdkit import Chem
from torch_geometric.data import Data
from openpom.models.mpnn_pom import MPNNPOMModel
from openpom.feat.graph_featurizer import GraphConvConstants

import deepchem as dc
from openpom.feat.graph_featurizer import GraphFeaturizer, GraphConvConstants
from openpom.utils.data_utils import get_class_imbalance_ratio
#from openpom.models.mpnn_pom import MPNNPOMModel
from datetime import datetime


class TritonPythonModel:
    def initialize(self, args):
        model_path = os.path.join(os.path.dirname(__file__), "example_model.pt")
        #self.model = torch.load(model_path, map_location=torch.device("cpu"))
        #self.model.eval()
        TASKS = [
'alcoholic', 'aldehydic', 'alliaceous', 'almond', 'amber', 'animal',
'anisic', 'apple', 'apricot', 'aromatic', 'balsamic', 'banana', 'beefy',
'bergamot', 'berry', 'bitter', 'black currant', 'brandy', 'burnt',
'buttery', 'cabbage', 'camphoreous', 'caramellic', 'cedar', 'celery',
'chamomile', 'cheesy', 'cherry', 'chocolate', 'cinnamon', 'citrus', 'clean',
'clove', 'cocoa', 'coconut', 'coffee', 'cognac', 'cooked', 'cooling',
'cortex', 'coumarinic', 'creamy', 'cucumber', 'dairy', 'dry', 'earthy',
'ethereal', 'fatty', 'fermented', 'fishy', 'floral', 'fresh', 'fruit skin',
'fruity', 'garlic', 'gassy', 'geranium', 'grape', 'grapefruit', 'grassy',
'green', 'hawthorn', 'hay', 'hazelnut', 'herbal', 'honey', 'hyacinth',
'jasmin', 'juicy', 'ketonic', 'lactonic', 'lavender', 'leafy', 'leathery',
'lemon', 'lily', 'malty', 'meaty', 'medicinal', 'melon', 'metallic',
'milky', 'mint', 'muguet', 'mushroom', 'musk', 'musty', 'natural', 'nutty',
'odorless', 'oily', 'onion', 'orange', 'orangeflower', 'orris', 'ozone',
'peach', 'pear', 'phenolic', 'pine', 'pineapple', 'plum', 'popcorn',
'potato', 'powdery', 'pungent', 'radish', 'raspberry', 'ripe', 'roasted',
'rose', 'rummy', 'sandalwood', 'savory', 'sharp', 'smoky', 'soapy',
'solvent', 'sour', 'spicy', 'strawberry', 'sulfurous', 'sweaty', 'sweet',
'tea', 'terpenic', 'tobacco', 'tomato', 'tropical', 'vanilla', 'vegetable',
'vetiver', 'violet', 'warm', 'waxy', 'weedy', 'winey', 'woody'
]
        input_file = '/opt/tritonserver/openpom/models/openpom_model/1/curated_GS_LF_merged_4983.csv' # or new downloaded file path
        featurizer = GraphFeaturizer()
        smiles_field = 'nonStereoSMILES'
        loader = dc.data.CSVLoader(tasks=TASKS,
                   feature_field=smiles_field,
                   featurizer=featurizer)
        self.dataset = loader.create_dataset(inputs=[input_file])
        dataset = loader.create_dataset(inputs=[input_file])
        n_tasks = len(dataset.tasks)

        self.model = MPNNPOMModel(
		    n_tasks = n_tasks,
            batch_size=128,
            learning_rate=0.001,
            loss_aggr_type='sum',
            node_out_feats=100,
            edge_hidden_feats=75,
            edge_out_feats=100,
            num_step_message_passing=5,
            mpnn_residual=True,
            message_aggregator_type='sum',
            mode='classification',
            number_atom_features=GraphConvConstants.ATOM_FDIM,
            number_bond_features=GraphConvConstants.BOND_FDIM,
            n_classes=1,
            readout_type='set2set',
            num_step_set2set=3,
            num_layer_set2set=2,
            ffn_hidden_list=[392, 392],
            ffn_embeddings=256,
            ffn_activation='relu',
            ffn_dropout_p=0.12,
            ffn_dropout_at_input_no_act=False,
            weight_decay=1e-5,
            self_loop=False,
            optimizer_name='adam',
            log_frequency=32,
            model_dir=model_path,  #
            device_name='cpu'
        )
        print("DEBUG: self.model type is", type(self.model))
#        self.model.eval()

    def execute(self, requests):
        responses = []

        for request in requests:
            try:

                smiles_input = pb_utils.get_input_tensor_by_name(request, "SMILES")
                smiles = smiles_input.as_numpy()  # shape: (batch, 1)
                smiles_str = [x[0].decode("utf-8") for x in smiles]
                
               
                graph_featurized = GraphFeaturizer()
                dataset = dc.data.NumpyDataset(X=[graph_featurized.featurize([smiles_str[0]])[0]])
                preds = self.model.predict(dataset)

                result = np.array(preds, dtype=np.float32)
                if result.ndim == 1:
                    result = result.reshape(-1, 1)  #
                output_tensor = pb_utils.Tensor("OUTPUT", result)
                inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])

                print("after predict")
                print("result: ", result)

            except Exception as e:
                error_response = pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(str(e))
                )
                responses.append(error_response)
                continue

            responses.append(inference_response)

        return responses
