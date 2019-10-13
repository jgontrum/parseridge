# Data preparation

This section describes the process of reading in treebanks and other input data, 
as well as highlighting the way this data represented internally.

## Treebank

```python
from parseridge.corpus.treebank import Treebank


treebank = Treebank(
    train_io=open("data/UD_English-GUM/en_gum-ud-train.conllu"),
    dev_io=open("data/UD_English-GUM/en_gum-ud-dev.conllu"),
    test_io=open("data/UD_English-GUM/en_gum-ud-test.conllu"),
)
```

The treebank class