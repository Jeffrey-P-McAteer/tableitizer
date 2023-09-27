
# Tableitizer

_Turn unstructured data into structured data through automated prompting of AI agents_

# Running

```bash
python -m tableitizer ./data/document1.txt ./data/cars_schema.json -v

# With a custom model
python -m tableitizer ./data/document1.txt ./data/cars_schema.json -v --model 'google/flan-t5-small' --model-source 'huggingface'

```

# Research

```bash
azure-contain tableitizer-container.toml
# Within container


```




