# Std Libs
import os
import sys
import argparse
import json

# Our Code
import tableitizer.experiments
import tableitizer.packages

json5 = tableitizer.packages.p_import('json5')

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
    import subprocess
    subprocess.run([
      sys.executable, '-m', 'ensurepip',
        '--user' #'--root', os.environ['TABLEITIZER_PY_ENV']
    ])
    import pip

def is_a_file(string):
  if not os.path.exists(string):
    raise Exception(f'The file {string} must exist!')
  if os.path.isdir(string):
    raise Exception(f'{string} must be a text file!')
  return string

def main(args=sys.argv):
  # Pop off args[0] if it's a python file
  if len(args) > 0 and args[0].lower().endswith('.py') and os.path.exists(args[0]):
    args = args[1:]
  
  parser = argparse.ArgumentParser(description='Tableitizer: Turn unstructured data into structured data through automated prompting of AI agents')
  parser.add_argument('doc', type=is_a_file)
  parser.add_argument('schema', type=is_a_file)
  parser.add_argument('csv_out', nargs='?', default='')
  
  args = parser.parse_args(args)

  set_py_env_folder()
  print(f'Using TABLEITIZER_PY_ENV = {os.environ.get("TABLEITIZER_PY_ENV", None)}')
  
  set_transformers_model_folder()
  print(f'Using TRANSFORMERS_CACHE = {os.environ.get("TRANSFORMERS_CACHE", None)}')
  
  doc_context = ''
  with open(args.doc, 'r') as fd:
    doc_context = fd.read()
    if not isinstance(doc_context, str):
      doc_context = doc_context.decode('utf-8')

  schema = {}
  with open(args.schema, 'r') as fd:
    schema = json5.load(fd)

  print(f'schema = {schema}')









