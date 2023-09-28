# Std Libs
import os
import sys
import argparse
from collections import namedtuple
import time
import json
import csv
import traceback
import re

# Our Code
import tableitizer.experiments
import tableitizer.packages

def set_transformers_model_folder():
  if not 'TRANSFORMERS_CACHE' in os.environ:
      os.environ['TRANSFORMERS_CACHE'] = os.path.join(
          os.getcwd(), '.tableitizer-model-weights'
      )
  os.makedirs(os.environ['TRANSFORMERS_CACHE'], exist_ok=True)
  os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# MUST be done before importing libs!!
set_transformers_model_folder()

# 3rd-party libs
json5 = tableitizer.packages.p_import('json5')

panml = tableitizer.packages.p_import('panml', 'git+https://github.com/Pan-ML/panml.git')
from panml.models import ModelPack
torch = tableitizer.packages.p_import('torch', 'torch torchvision torchaudio')

# Optional/addtl LLM optimization packages. Required for load_in_8bit=True in model load params.
try:
  accelerate = tableitizer.packages.p_import('accelerate')
except Exception:
  pass
try:
  bitsandbytes = tableitizer.packages.p_import('bitsandbytes')
except Exception:
  pass

try:
  word2number = tableitizer.packages.p_import('word2number')
except Exception:
  pass
from word2number import w2n

def is_a_file(string):
  if not os.path.exists(string):
    raise Exception(f'The file {string} must exist!')
  if os.path.isdir(string):
    raise Exception(f'{string} must be a text file!')
  return string

def gen_table_reader(schema_key, schema_val):
  table_reader = namedtuple('TableReader', 'table_name table_count_queries row_field_query_dict')
  if not ':=' in schema_key:
    raise Exception(f'Schema key "{schema_key}" must have ":=" between the table name and the query for number of items!')

  schema_key_parts = schema_key.split(':=')
  table_reader.table_name = schema_key_parts[0].strip().lower()
  table_reader.table_count_queries = []
  for query_text in (' '.join(schema_key_parts[1:]).strip()).split(';;'):
    table_reader.table_count_queries.append(query_text.strip())
  table_reader.row_field_query_dict = dict()
  table_reader.row_field_parser_fn = dict()

  if isinstance(schema_val, list) and len(schema_val) > 1:
    raise Exception(f'Schema values MAY be lists for structure similarity, but if so they must have only a single dictionary of key prompts! You have given the library {len(schema_val)} items.')

  if isinstance(schema_val, list) and len(schema_val) == 1:
    schema_val = schema_val[0]

  if not isinstance(schema_val, dict):
    raise Exception(f'Schema values must be dictionaries, got {schema_val}.')

  for k, v in schema_val.items():
    if ':=' in k:
      key = k.split(':=')[0].lower().strip()
      parse_fn = k.split(':=')[1].strip()

      if parse_fn in globals():
        table_reader.row_field_parser_fn[key] = globals()[parse_fn]
      elif parse_fn in locals():
        table_reader.row_field_parser_fn[key] = locals()[parse_fn]
        

      table_reader.row_field_query_dict[key] = v
    else:
      table_reader.row_field_query_dict[k.lower().strip()] = v

  return table_reader

NON_DECIMAL_REGEX = re.compile(r'[^\d.]+')
NON_INT_REGEX = re.compile(r'[^\d]+')

def parse_int(text):
  if text is None:
    return None
  text = text.lower().strip()
  try:
    integer_text = NON_INT_REGEX.sub('', text).strip() # remove all except [decimal] chars
    return int(integer_text)
  except Exception:
    try:
      return int(w2n.word_to_num(text))
    except Exception:
      return None

def parse_float(text):
  if text is None:
    return None
  text = text.lower().strip()
  try:
    float_text = NON_DECIMAL_REGEX.sub('', text).strip() # remove all except [decimal + '.'] chars
    return float(float_text)
  except Exception:
    try:
      return float(w2n.word_to_num(text))
    except Exception:
      return None

def parse_str(text):
  if text is None:
    return None
  return str(text)

def answer_questions(lm, doc_context, questions, parser_fn, verbosity=0):
  if not isinstance(questions, list):
    questions = [ questions ]

  for question in questions:
    # See https://github.com/google-research/FLAN/blob/main/flan/v2/flan_templates_branched.py#L277C13-L277C13
    prompt_templates = [
      f'''Article: {doc_context}\n\nQuestion: {question}'''.strip(),
      f'''Read this and answer the question\n\n{doc_context}\n\n{question}'''.strip(),
      f'''Article: {doc_context}\n\nNow answer this question: {question}'''.strip(),
    ]
    for prompt_template in prompt_templates:
      
      if verbosity > 2:
        print()
        llm_input = prompt_template.replace(doc_context, '{doc_context}').replace('\n\n', '\n').strip()
        llm_input = llm_input.replace('\n', '\nLLM>>> ')
        print(f'LLM>>> {llm_input}')
      
      lm_output = lm.predict(prompt_template, max_length=2048, keep_history=False)

      if verbosity > 2:
        print(f'LLM<<< {lm_output.get("text", None)}')
        if lm_output.get("text", None) is None:
          print(f'EXT<<< lm_output = {json.dumps(lm_output, indent=2)}')

      lm_response = None
      if 'text' in lm_output:
        try:
          lm_out_text = lm_output['text']
          if prompt_template in lm_out_text:
            # We got a model that repeats inputs w/ answers as extensions
            lm_out_text = lm_out_text.replace(prompt_template, '').strip()
          lm_response = parser_fn(lm_out_text)
        except Exception:
          traceback.print_exc()
      else:
        lm_response = json.dumps(lm_output)
      
      if lm_response is not None:
        return lm_response
    
  return None



def main(args=sys.argv):
  # Pop off args[0] if it's a python file
  if len(args) > 0 and args[0].lower().endswith('.py') and os.path.exists(args[0]):
    args = args[1:]
  model_args_dir = os.path.join(os.path.dirname(__file__), 'model_args')
  
  ## 
  ## First we read all cli args (or could be called from another module; import tableitizer, tableitizer.main(['file.txt' ... ]))
  ##

  parser = argparse.ArgumentParser(description='Tableitizer: Turn unstructured data into structured data through automated prompting of AI agents')
  parser.add_argument('doc', type=is_a_file)
  parser.add_argument('schema', type=is_a_file)
  parser.add_argument('out_file', nargs='?', default='', help='''
Output data file; if name ends in .json, all tables will be placed in one file.
If name ends in .csv AND >1 table is defined, multiple <prefix>_<table-name>.csv
files will be output. All .csv files will contain 1 row of header names.
'''.strip())
  parser.add_argument('-v', '--verbose', action='count', default=0)
  parser.add_argument('--model', nargs='?', default='google/flan-t5-large', help='''
Example models:
  - bert-base-uncased    -110m parameters
  - bert-large-uncased   -340m parameters
  - google/flan-t5-base  -? parameters, needs 3gb ram
  - google/flan-t5-large -? parameters, needs 4gb ram
  - bigscience/bloomz-1b7 - 1b parameters, needs 8gb ram

'''.strip())
  parser.add_argument('--model-source', nargs='?', default='huggingface')
  parser.add_argument('--model-args-dir', nargs='?', default=model_args_dir, help='''
Location of model-specific arguments used during loading. File names must be the
last token of the model name; for example the model "bert-base-uncased" would have
configuration loaded from "bert-base-uncased.json" if that file exists,
and the model "google/flan-t5-base" would have configuration loaded
from "flan-t5-base.json" if that file exists.
'''.strip())

  args = parser.parse_args(args)

  ## 
  ## Now read argument data from files into useful data structures
  ##

  if args.verbose > 0:
    try:
      print(f'torch.cuda.is_available() = {torch.cuda.is_available()}')
      print(f'torch.backends.mps.is_available() = {torch.backends.mps.is_available()}')
    except Exception:
      traceback.print_exc()

  set_transformers_model_folder()
  if args.verbose > 0:
    print(f'Using TRANSFORMERS_CACHE = {os.environ.get("TRANSFORMERS_CACHE", None)}')
  
  doc_context = ''
  with open(args.doc, 'r') as fd:
    doc_context = fd.read()
    if not isinstance(doc_context, str):
      doc_context = doc_context.decode('utf-8')

  schema = {}
  with open(args.schema, 'r') as fd:
    schema = json5.load(fd)

  if args.verbose > 0:
    print(f'schema = {schema}')

  if not isinstance(schema, dict):
    raise Exception(f'The schema data in {args.schema} is not a dictionary! Please use a dictinary as the top-level object in schemas.')

  table_readers = []
  for k, v in schema.items():
    table_readers.append(
      gen_table_reader(k, v)
    )

  if args.verbose > 1:
    print(f'table_readers = {table_readers}')
    for t in table_readers:
      print()
      print(f't.table_name={t.table_name}')
      print(f't.table_count_queries={t.table_count_queries}')
      print(f't.row_field_query_dict={t.row_field_query_dict}')

  ## 
  ## Load the LLM
  ##

  known_good_model_args = dict()
  if os.path.exists(model_args_dir):
    for model_arg_file in os.listdir(model_args_dir):
      if not model_arg_file.lower().endswith('.json'):
        continue
      model_arg_name = os.path.basename(model_arg_file).lower().replace('.json', '')
      try:
        with open(model_arg_file, 'r') as fd:
          known_good_model_args[model_arg_name] = json5.load(fd)
      except:
        traceback.print_exc()

  if args.verbose > 0:
    print(f'Loading model {args.model} from {args.model_source}')
  model_load_begin_s = time.perf_counter()

  model_name_token = os.path.basename(args.model)
  if args.verbose > 1:
    print(f'Model load is using the following args: {json.dumps(known_good_model_args.get(model_name_token, dict()), indent=2)}')

  lm = ModelPack(
    model=args.model,
    source=args.model_source,
    model_args=known_good_model_args.get(model_name_token, dict())
  )
  # ^^ See model_args extras passed into .from_pretrained() https://huggingface.co/transformers/v3.5.1/model_doc/auto.html#automodelforseq2seqlm


  model_load_end_s = time.perf_counter()
  if args.verbose > 0:
    print(f'{args.model} loaded in {model_load_end_s - model_load_begin_s:0.1f}s')

  ## 
  ## Parse all the data using the llm!
  ##

  parsed_data = dict()
  for t in table_readers:
    if args.verbose > 0:
      print('=' * 6, t.table_name, '=' * 6)

    num_items = answer_questions(lm, doc_context, t.table_count_queries, parse_int, verbosity=args.verbose)
    if args.verbose > 0:
      print(f'{t.table_name} num_items = {num_items}')
      
    if num_items is None:
      continue

    parsed_data[t.table_name] = list()
    for item_num in range(1, num_items+1):
      try:
        row_vals = dict()
        for field_name, field_questions in t.row_field_query_dict.items():
          if not isinstance(field_questions, list):
            field_questions = [ field_questions ]
          field_questions = field_questions.copy() # Duplicate list so we don't write back to t.row_field_query_dict

          for field_questions_i in range(0, len(field_questions)):
            if '{' in field_questions[field_questions_i] and '}' in field_questions[field_questions_i]:
              field_questions[field_questions_i] = field_questions[field_questions_i].format(item_num)

          if args.verbose > 1:
            print(f'row_vals[{field_name}] = answer_questions(lm, doc_context, {field_questions}, {t.row_field_parser_fn.get(field_name, parse_str)})')

          row_vals[field_name] = answer_questions(lm, doc_context, field_questions, t.row_field_parser_fn.get(field_name, parse_str), verbosity=args.verbose )

        parsed_data[t.table_name].append(row_vals)
      except Exception:
        traceback.print_exc()


  ## 
  ## Now write results someplace useful!
  ##

  if not args.out_file:
    # Args file is falsey, print everything to stdout
    print('No out_file specified, printing all data to stdout!')
    for table_name, table_rows in parsed_data.items():
      print('=' * 6, table_name, '=' * 6)
      for row in table_rows:
        print(json.dumps(row, indent=2))
      print()

  else:
    if len(parsed_data) < 1:
      print(f'No tables specified, cannot write anything to {args.out_file}!')
      return

    if args.out_file.lower().endswith('.json'):
      print(f'Writing results to {args.out_file}')
      with open(args.out_file, 'w') as fd:
        json.dump(parsed_data, fd, indent=2)

    elif args.out_file.lower().endswith('.csv'):
      if len(parsed_data) > 1:
        out_dir_name = os.path.dirname(os.path.abspath(args.out_file))
        filename_base, extension = os.path.splitext(os.path.basename(args.out_file))

        for table_name, rows in parsed_data.items():
          out_csv_file = os.path.join(out_dir_name, f'{filename_base}_{table_name}{extension}')
          print(f'Writing table {table_name} to {out_csv_file}')
          
          col_names = list()
          for row_dict in parsed_data[table_name]:
            for col_name in row_dict.keys():
              if not col_name in col_names:
                col_names.append(col_name)

          with open(out_csv_file, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(col_names)
            for row_dict in parsed_data[table_name]:
              row_vals = []
              for col_name in col_names:
                row_vals.append(row_dict.get(col_name, '')) # Get value || empty string
              csvwriter.writerow(row_vals)
        
      else:
        table_name = list(parsed_data.keys())[0]
        print(f'Writing table {table_name} to {args.out_file}')
        
        col_names = list()
        for row_dict in parsed_data[table_name]:
          for col_name in row_dict.keys():
            if not col_name in col_names:
              col_names.append(col_name)

        with open(args.out_file, 'w', newline='') as csvfile:
          csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
          csvwriter.writerow(col_names)
          for row_dict in parsed_data[table_name]:
            row_vals = []
            for col_name in col_names:
              row_vals.append(row_dict.get(col_name, '')) # Get value || empty string
            csvwriter.writerow(row_vals)
      
    else:
      print(f'Unknown extension for {args.out_file}, writing JSON-formatted data to it!')
      with open(args.out_file, 'w') as fd:
        json.dump(parsed_data, fd, indent=2)










