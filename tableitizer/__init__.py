# Std Libs
import os
import sys
import argparse

# Our Code
import tableitizer.experiments

def set_transformers_model_folder():
  if not 'TRANSFORMERS_CACHE' in os.environ:
      os.environ['TRANSFORMERS_CACHE'] = os.path.join(
          os.getcwd(), '.tableitizer-model-weights'
      )
  os.makedirs(os.environ['TRANSFORMERS_CACHE'], exist_ok=True)

def set_py_env_folder():
  if not 'TABLEITIZER_PY_ENV' in os.environ:
    os.environ['TABLEITIZER_PY_ENV'] = os.path.join(os.getcwd(), '.tableitizer-py-env')
  os.makedirs(os.environ['TABLEITIZER_PY_ENV'], exist_ok=True)
  sys.path.append(os.environ['TABLEITIZER_PY_ENV'])
  os.environ['PYTHONPATH'] = os.environ['TABLEITIZER_PY_ENV']+os.pathsep+os.environ.get('PYTHONPATH', '')
  # Also ensure we have pip in sys packags OR in TABLEITIZER_PY_ENV
  try:
    import pip
  except:
    import ensurepip
    ensurepip.main([
      '--root', os.environ['TABLEITIZER_PY_ENV']
    ])
    import pip


def main(args=sys.argv):
  parser = argparse.ArgumentParser(description='Tableitizer: Turn unstructured data into structured data through automated prompting of AI agents')


  args = parser.parse_args(args[1:])

  set_py_env_folder()
  print(f'Using TABLEITIZER_PY_ENV = {os.environ.get("TABLEITIZER_PY_ENV", None)}')
  
  set_transformers_model_folder()
  print(f'Using TRANSFORMERS_CACHE = {os.environ.get("TRANSFORMERS_CACHE", None)}')
  
  print(f'args={args}')

  tableitizer.experiments.simple_model_poc()





