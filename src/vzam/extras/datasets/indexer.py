import os
import json
import faiss
import pandas as pd
import logging
import copy
from pathlib import Path
from typing import Any, Dict
from kedro.io import AbstractDataSet
from torchvision.datasets import ImageFolder

from vzam.indexer import Indexer

log = logging.getLogger(__name__)


class IndexerDataset(AbstractDataSet):
    def __init__(
        self,
        filepath: str,
        # quantizer_fname: str,
        index_fname: str,
        args_fname: str,
        load_args: Dict[str, Any] = None,
        save_args: Dict[str, Any] = None,
    ) -> None:
        """
        Args:
            filepath:
            load_args:
            save_args:
        """

        super().__init__()
        self._filepath = filepath
        # self._quantizer_fname = quantizer_fname
        self._index_fname = index_fname
        self._args_fname = args_fname

        # self._quantizer_path = os.path.join(self._filepath, self._quantizer_fname)
        self._index_path = os.path.join(self._filepath, self._index_fname)
        self._args_path = os.path.join(self._filepath, self._args_fname)
        self._load_args = load_args
        self._save_args = save_args

    def _load(self) -> Any:
        index = faiss.read_index(self._index_path)
        with open(self._args_path, 'r') as f:
            kwargs = json.load(f)
        indexer = Indexer(index=index, **kwargs)
        return indexer

    def _save(self, indexer) -> None:
        if not os.path.exists(self._filepath):
            os.makedirs(self._filepath)

        faiss.write_index(indexer.index, self._index_path)
        with open(self._args_path, 'w') as f:
            json.dump({
                'nlist': indexer.nlist,
                'd': indexer.d,
                'metadata': indexer.metadata
            }, f)

    def _describe(self) -> Dict[str, Any]:
        return dict(
            filepath=self._filepath,
            index_path=self._index_path,
            args_path=self._args_path,
            load_args=self._save_args,
            save_args=self._save_args,
        )
