import os
import sys

def simple_falcon_responses():
  try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    import transformers
    import torch
    import safetensors
    # raise Exception('manual re-install of dependencies')
  except:
    import pip
    pip.main([
      'install', f'--target={os.environ.get("TABLEITIZER_PY_ENV", None)}',
        'torch>=2.0.1',
        'transformers',
        'einops',
        'accelerate',
        'safetensors',
        'hf-hub-ctranslate2>=2.0.8',
        'ctranslate2>=3.14.0'
    ])

    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    import transformers
    import torch
    import safetensors

  model = 'tiiuae/falcon-rw-1b'

  tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, use_fast=True)
  config = AutoConfig.from_pretrained(model, trust_remote_code=True)

  pipeline = transformers.pipeline(
      'text-generation',
      model=model,
      tokenizer=tokenizer,
      torch_dtype=torch.bfloat16,
      trust_remote_code=True,
      device_map='auto',
      # offload_folder=os.environ['TRANSFORMERS_CACHE']
  )

  initial_docs_and_prompt = f'''
There are four cars on the road.
The first car is colored rec and has a sunroof.
The second car is colored blue.
The third car is colored yellow.
The fifth car is colored green.
The sixth car on the road is white and smells of oil.

'''.strip() +'\n\n'+ '''
How many cars are on the road?
'''.strip()

  sequences = pipeline(
      initial_docs_and_prompt,
      max_length=850,
      do_sample=True,
      top_k=10,
      num_return_sequences=1,
      eos_token_id=tokenizer.eos_token_id,
  )

  print('=' * 8, 'prompt', '=' * 8)
  print(initial_docs_and_prompt);
  print('=' * 8, '======', '=' * 8)
  

  for seq in sequences:
      print(f"Result: {seq['generated_text']}")

  print(f'pipeline = {pipeline}')

  # TODO
  import code
  v = globals()
  v.update(locals())
  code.interact(local=v)


def simple_model_poc():
  try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    import transformers
    import torch
    import safetensors
    # raise Exception('manual re-install of dependencies')
  except:
    import pip
    pip.main([
      'install', f'--target={os.environ.get("TABLEITIZER_PY_ENV", None)}',
        'torch>=2.0.1',
        'transformers',
        'einops',
        'accelerate',
        'safetensors',
        'hf-hub-ctranslate2>=2.0.8',
        'ctranslate2>=3.14.0'
    ])

    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    import transformers
    import torch
    import safetensors

  #model = 'tiiuae/falcon-7b-instruct'
  #model = 'vilsonrodrigues/falcon-7b-instruct-sharded'
  model = 'tiiuae/falcon-rw-1b'

  tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, use_fast=True)
  config = AutoConfig.from_pretrained(model, trust_remote_code=True)

  pipeline = transformers.pipeline(
      'text-generation',
      model=model,
      tokenizer=tokenizer,
      torch_dtype=torch.bfloat16,
      trust_remote_code=True,
      device_map='auto',
      offload_folder=os.environ['TRANSFORMERS_CACHE']
  )
  sequences = pipeline(
     "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:",
      max_length=200,
      do_sample=True,
      top_k=10,
      num_return_sequences=1,
      eos_token_id=tokenizer.eos_token_id,
  )
  for seq in sequences:
      print(f"Result: {seq['generated_text']}")

def langchain_poc():
  try:
    import langchain
    import gpt4all
    from transformers import AutoModelForCausalLM
  except:
    import pip
    pip.main([
      'install', f'--target={os.environ.get("TABLEITIZER_PY_ENV", None)}',
        'langchain[all]',
        'gpt4all',

        'torch>=2.0.1',
        'transformers',
        'einops',
        'accelerate',
        'safetensors',
    ])
    import langchain
    import gpt4all
    from transformers import AutoModelForCausalLM

  # Grab a model
  model = AutoModelForCausalLM.from_pretrained("nomic-ai/gpt4all-j")
  print(f'model = {model}')

  # Grab a local model type
  from langchain.llms import GPT4All
  llm = GPT4All(model=
    os.path.join(os.environ['TRANSFORMERS_CACHE'], "nous-hermes-13b.ggmlv3.q4_0.bin")
  )
  r = llm("The first man on the moon was ... Let's think step by step")
  print(f'r = {r}')






