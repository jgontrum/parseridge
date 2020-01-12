from collections import defaultdict
from typing import TypeVar, Generic, Optional, List, Dict

from parseridge.utils.logger import LoggerMixin

T = TypeVar("T")


class Signature(LoggerMixin, Generic[T]):
    def __init__(
        self, entries: Optional[List[T]] = None, warn_on_oov: bool = False
    ) -> None:
        self._read_only = False
        self._warn_on_oov = warn_on_oov

        self._item_to_id: Dict[T, int] = {}
        self._id_to_item: List[T] = []
        self._item_to_count: Dict[T, int] = defaultdict(int)
        self.oov = 0  # The default id of the out-of-vocabulary item

        # Add default entries like 'Padding', 'OOV' etc.
        if entries is not None:
            for item in entries:
                self.add(item)

    def read_only(self) -> None:
        """
        Makes this object read-only to prevent accidental changes to the mapping.
        """
        self._read_only = True

    def add(self, item: T) -> int:
        """
        Adds an item to the mapping and assigns it an id.
        If the item already exists, the id of the item is returned.
        Parameters
        ----------
        item : The item to add.

        Returns
        -------
        int : The id of the added item.
        """
        if self._read_only:
            id_ = self._item_to_id.get(item, self.oov)
            if id_ == self.oov and self._warn_on_oov:
                self.logger.warning(f"Item not found: '{item}'")
            return id_

        self._item_to_count[item] += 1

        token_id = self._item_to_id.get(item)
        if token_id is None:
            token_id = len(self._id_to_item)
            self._item_to_id[item] = token_id
            self._id_to_item.append(item)

        return token_id

    def get_id(self, item: T) -> int:
        """
        Returns the id of the given item, if it exists. Otherwise, the OOV-id is returned.
        Parameters
        ----------
        item : The item to get the id of.

        Returns
        -------
        int : The id of the given item.
        """
        return self._item_to_id.get(item, self.oov)

    def get_count(self, item: T) -> int:
        """
        Returns how often the given item has been added to the mapping.

        Parameters
        ----------
        item : The item to get the counts of.

        Returns
        -------
        int : How often the item was seen.
        """
        return self._item_to_count.get(item, 0)

    def get_item(self, id_: int) -> T:
        """
        Returns the item for the given id.
        Parameters
        ----------
        id_ : The id of the item in question.

        Returns
        -------
        T : The item for the given id.
        """
        if id_ < len(self._id_to_item):
            return self._id_to_item[id_]
        if self._warn_on_oov:
            self.logger.warning(f"Item not found: '{id_}'")
        return self._id_to_item[self.oov]

    def get_items(self) -> List[T]:
        """
        Returns a list of all items in this signature

        Returns
        -------
        List of items in the signature.
        """
        return self._id_to_item

    def __len__(self) -> int:
        return len(self._id_to_item)

    def __repr__(self) -> str:
        return f"Vocabulary size: {len(self)}"
