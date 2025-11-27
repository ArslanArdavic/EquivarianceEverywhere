from enum import Enum, auto
from typing import List
import math

from helpers.datasets import DataSet

TRAIN_MULTISETS = [
        [],
        [DataSet.texas, DataSet.tolokers],
        [DataSet.photo, DataSet.texas, DataSet.roman_empire, DataSet.tolokers],
        [DataSet.photo, DataSet.texas, DataSet.usa, DataSet.actor, DataSet.roman_empire, DataSet.tolokers],
        [DataSet.computers, DataSet.photo, DataSet.texas, DataSet.usa, DataSet.europe, DataSet.actor, DataSet.roman_empire, DataSet.tolokers],
        [DataSet.roman_empire, DataSet.amazon_ratings, DataSet.minesweeper, DataSet.tolokers, DataSet.questions, DataSet.pubmed, DataSet.citeseer, DataSet.chameleon, DataSet.squirrel, DataSet.cornell, DataSet.wisconsin, DataSet.texas, DataSet.full_cora, DataSet.full_DBLP, DataSet.wiki_attr, DataSet.blogcatalog, DataSet.wiki_cs, DataSet.co_cs, DataSet.co_physics, DataSet.usa, DataSet.europe, DataSet.actor, DataSet.computers, DataSet.photo, DataSet.deezer, DataSet.arxiv],
    ]


class TrainTestSetup(Enum):
    trainset1 = auto()
    inc_trainset = auto()

    _all_datasets = set(DataSet)

    @staticmethod
    def from_string(s: str):
        try:
            return TrainTestSetup[s]
        except KeyError:
            raise ValueError(f"Unknown setup name: {s}")

    def get_train_datasets(self, train_size: int) -> List[DataSet]:
        if self is TrainTestSetup.trainset1:
            return [DataSet.cora]

        assert train_size in [1, 3, 5, 7, 9, 27], "Invalid train size"
        train_idx = math.floor(train_size/2)
        if train_size == 27:
            train_idx = 5
        trainset_list = [DataSet.cora] + TRAIN_MULTISETS[train_idx]
        return sorted(trainset_list, key=lambda x: x.value)

    def get_test_datasets(self) -> List[DataSet]:
        trainset = {DataSet.cora}
        if self is TrainTestSetup.inc_trainset:
            trainset.update(TRAIN_MULTISETS[-1])
        testset_list = list(set(DataSet) - trainset)
        return sorted(testset_list, key=lambda x: x.value)
