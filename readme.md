
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
# Spawns a bash shell
azure-contain tableitizer-container.toml

# Runs a big model under constraints & writes some output to out/
azure-contain tableitizer-container.toml python3.10 -m tableitizer ./data/document1.txt ./data/cars_multi_schema.json out/test.csv -vvv --model google/flan-t5-xl
```




