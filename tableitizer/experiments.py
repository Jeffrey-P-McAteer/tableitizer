

def simple_model_poc():
  try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import transformers
    import torch
  except:
    import os
    import pip
    pip.main([
      'install', f'--target={os.environ.get("TABLEITIZER_PY_ENV", None)}',
        'torch==2.0.1',
        'transformers',
        'einops',
        'accelerate',
    ])

    from transformers import AutoTokenizer, AutoModelForCausalLM
    import transformers
    import torch

  model = "tiiuae/falcon-7b-instruct"

  tokenizer = AutoTokenizer.from_pretrained(model)
  pipeline = transformers.pipeline(
      "text-generation",
      model=model,
      tokenizer=tokenizer,
      torch_dtype=torch.bfloat16,
      trust_remote_code=True,
      device_map="auto",
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


