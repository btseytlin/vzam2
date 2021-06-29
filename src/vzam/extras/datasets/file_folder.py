import os
import pandas as pd
import logging
import copy
from pathlib import Path
from typing import Any, Dict
from kedro.io import AbstractDataSet
from torchvision.datasets import ImageFolder

log = logging.getLogger(__name__)


class FileFolderDataSet(AbstractDataSet):
    """Loads a folder containing videos as an iterable"""

    def __init__(
        self,
        filepath: str,
        dir_name: str,
        load_args: Dict[str, Any] = None,
        save_args: Dict[str, Any] = None,
    ) -> None:
        """
        Args:
            filepath:
            load_args:
            save_args: Ignored as saving is not supported.
            version: If specified, should be an instance of
                ``kedro.io.core.Version``. If its ``load`` attribute is
                None, the latest version will be loaded.
        """

        super().__init__()
        self._filepath = filepath
        self._dir_name = dir_name
        self._dir_path = os.path.join(self._filepath, self._dir_name)
        self._load_args = load_args
        self._save_args = save_args

    def _load(self) -> Any:
        return os.scandir(self._dir_path)

    def _save(self, video_folder) -> None:
        """ Not Implemented """
        pass

    def _describe(self) -> Dict[str, Any]:
        return dict(
            filepath=self._filepath,
            dir_path=self._dir_path,
            load_args=self._save_args,
            save_args=self._save_args,
        )
