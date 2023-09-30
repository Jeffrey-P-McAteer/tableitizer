
# Tableitizer

_Turn unstructured data into structured data through automated prompting of AI agents_

# Running

```bash
python -m tableitizer ./data/document1.txt ./data/cars_schema.json -v

# With a custom model
python -m tableitizer ./data/document1.txt ./data/cars_schema.json -v --model 'google/flan-t5-small' --model-source 'huggingface'

# bloomz-1b7 uses a ton of memory but performance is great! It can capture
# referential statements like "The Jeep is facing the same direction as the second car" and backtrack to the data referenced.
python -m tableitizer ./data/document1.txt ./data/cars_multi_schema.json -vvv --model bigscience/bloomz-1b7

# Initial 2+ model research
python -m tableitizer ./data/document1.txt ./data/cars_multi_schema.json -vvv  --model google/flan-t5-large --model bigscience/bloomz-1b7

# Same but larger models
python -m tableitizer ./data/document1.txt ./data/cars_multi_schema.json -vvv  --model google/flan-t5-xl --model bigscience/bloomz-3b

```

# Research

```bash
# Spawns a bash shell
azure-contain tableitizer-container.toml

# Runs a big model under constraints & writes some output to out/
azure-contain tableitizer-container.toml python3.10 -m tableitizer ./data/document1.txt ./data/cars_multi_schema.json out/test.csv -vvv --model google/flan-t5-xl
```




