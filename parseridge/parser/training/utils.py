from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from parseridge.corpus.training_data import ConLLDataset, ConfigurationGenerator
from parseridge.utils.helpers import Transition


def conll_data_set_to_data_frame(conll_data_set: ConLLDataset) -> pd.DataFrame:
    get_relation_str = conll_data_set.data_generator.relations.label_signature.get_item
    get_transition_str = Transition.get_item
    get_token_str = conll_data_set.data_generator.vocabulary.get_item

    dicts = []
    for data_point in tqdm(conll_data_set.data_points, desc="Building data frame..."):
        data_point_dict = asdict(data_point)
        data_point_dict = {k: v.numpy() for k, v in data_point_dict.items()}

        sentence_str = [get_token_str(id_) for id_ in data_point.sentence.numpy()]
        data_point_dict.update(
            {
                "sentence_str": sentence_str,
                "stack_str": [sentence_str[idx] for idx in data_point.stack.numpy()],
                "buffer_str": [sentence_str[idx] for idx in data_point.buffer.numpy()],
                "gold_transition_str": get_transition_str(
                    data_point.gold_transition.numpy()
                ),
                "gold_relation_str": get_relation_str(data_point.gold_relation.numpy())[1],
                "wrong_transitions_str": [
                    get_transition_str(i) for i in data_point.wrong_transitions.numpy()
                ],
                "wrong_relations_str": [
                    get_relation_str(i)[1] for i in data_point.wrong_relations.numpy()
                ],
            }
        )

        dicts.append(data_point_dict)

    return pd.DataFrame(dicts)


def data_frame_to_conll_data_points_(
    conll_data_set: ConLLDataset, df: pd.DataFrame
) -> ConLLDataset:
    data_points = []

    for record in tqdm(df.to_dict("records"), desc="Converting to ConLLDataset..."):
        clean_record = {
            k: torch.from_numpy(v) for k, v in record.items() if isinstance(v, np.ndarray)
        }

        data_points.append(ConfigurationGenerator.ConfigurationItem(**clean_record))

    conll_data_set.data_points = data_points

    return conll_data_set
