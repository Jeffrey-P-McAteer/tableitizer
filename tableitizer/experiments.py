import os
import sys

def flan_experiment():
  model_to_use = 'google/flan-t5-xl' # 3b parameter model, uses maybe 12gb ram.
  model_source = 'huggingface'

  # See https://github.com/Pan-ML/panml
  try:
      import panml
  except:
      import pip
      pip.main([
        'install', f'--target={os.environ.get("TABLEITIZER_PY_ENV", None)}',
          'git+https://github.com/Pan-ML/panml.git'
      ])
      import panml

  from panml.models import ModelPack

  lm = ModelPack(model=model_to_use, source=model_source, model_args={'gpu': True})

  def lm_answer(context_text, question_text):
    lm_output = lm.predict(f'''
Given the following document:
===
{context_text}
===
Answer this question correctly: "{question_text}"
'''.strip(), max_length=2048)
    return lm_output['text']


  context_text = '''
There are four cars on the road.
The first car is colored rec and has a sunroof.
The second car is colored blue.
The third car is colored yellow.
The fifth car is colored green.
The sixth car on the road is white and smells of oil.
  '''.strip()

  question_text = '''
How many cars are visible on the road?
  '''.strip()

  print(f'lm = {lm}')
  print(f'lm_answer(context_text, question_text)')
  print(f'lm_answer("words words words", "How many words were spoken?")')


  import code
  v = globals()
  v.update(locals())
  code.interact(local=v)


def simple_falcon_responses():
  import re
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

  #model = 'tiiuae/falcon-rw-1b'
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
There are currently this many cars on the road: 
'''.strip()

  sequences = pipeline(
      initial_docs_and_prompt,
      max_length=250,
      do_sample=True,
      #top_k=10,
      num_return_sequences=1,
      eos_token_id=tokenizer.eos_token_id,
  )

  print('=' * 8, 'prompt', '=' * 8)
  print(initial_docs_and_prompt);
  print('=' * 8, '======', '=' * 8)
  
  # Correct answer is 5, but this is kind of a tricky Q for a low-level model.
  nums = dict()
  for seq in sequences:
      print(f"Result: {seq['generated_text']}")
      try:
        for possible_num_txt in re.findall(r'\d+', seq['generated_text']):
          possible_num = int(possible_num_txt)
          if not possible_num in nums:
            nums[possible_num] = 0
          if 'cars' in possible_num_txt:
            nums[possible_num] += 10
          else:
            nums[possible_num] += 1
      except:
        pass


  print(f'pipeline = {pipeline}')
  print(f'nums = {nums} (we want to see 5 in here!)')

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






