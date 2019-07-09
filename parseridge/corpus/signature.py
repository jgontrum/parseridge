from collections import defaultdict

from parseridge.utils.logger import LoggerMixin


class Signature(LoggerMixin):
    def __init__(self, entries=None, warn_on_oov=False):
        self._read_only = False
        self._warn_on_oov = warn_on_oov

        self._item_to_id = {}
        self._id_to_item = []
        self._item_to_count = defaultdict(int)
        self.oov = 0

        # Add default entries like 'Padding', 'OOV' etc.
        if entries is not None:
            for item in entries:
                self.add(item)

    def read_only(self):
        """
        Prevents further changes to the data and returns the OOV token if
        unknown entries are requested.
        :return:
        """
        self._read_only = True

    def add(self, item):
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

    def get_id(self, item):
        return self._item_to_id.get(item, self.oov)

    def get_count(self, item):
        return self._item_to_count.get(item, 0.0)

    def get_item(self, id_):
        if id_ < len(self._id_to_item):
            return self._id_to_item[id_]
        if self._warn_on_oov:
            self.logger.warning(f"Item not found: '{id_}'")
        return self._id_to_item[self.oov]

    def get_items(self):
        return self._id_to_item

    def __len__(self):
        return len(self._id_to_item)

    def __repr__(self):
        return f"Vocabulary size: {len(self)}"
