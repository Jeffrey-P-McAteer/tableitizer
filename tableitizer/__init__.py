# Std Libs
import os
import sys
import argparse
from collections import namedtuple
import time
import json
import traceback
import re

# Our Code
import tableitizer.experiments
import tableitizer.packages

# 3rd-party libs
json5 = tableitizer.packages.p_import('json5')

panml = tableitizer.packages.p_import('panml', 'git+https://github.com/Pan-ML/panml.git')
from panml.models import ModelPack

try:
  word2number = tableitizer.packages.p_import('word2number')
except:
  pass
from word2number import w2n


def set_transformers_model_folder():
  if not 'TRANSFORMERS_CACHE' in os.environ:
      os.environ['TRANSFORMERS_CACHE'] = os.path.join(
          os.getcwd(), '.tableitizer-model-weights'
      )
  os.makedirs(os.environ['TRANSFORMERS_CACHE'], exist_ok=True)

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
  except:
    try:
      return int(w2n.word_to_num(text))
    except:
      return None

def parse_float(text):
  if text is None:
    return None
  text = text.lower().strip()
  try:
    float_text = NON_DECIMAL_REGEX.sub('', text).strip() # remove all except [decimal + '.'] chars
    return float(float_text)
  except:
    try:
      return float(w2n.word_to_num(text))
    except:
      return None

def parse_str(text):
  if text is None:
    return None
  return str(text)

def answer_questions(lm, doc_context, questions, parser_fn):
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
      lm_output = lm.predict(prompt_template, max_length=2048)
      lm_response = None
      if 'text' in lm_output:
        try:
          lm_response = parser_fn(lm_output['text'])
        except:
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
  
  parser = argparse.ArgumentParser(description='Tableitizer: Turn unstructured data into structured data through automated prompting of AI agents')
  parser.add_argument('doc', type=is_a_file)
  parser.add_argument('schema', type=is_a_file)
  parser.add_argument('csv_out', nargs='?', default='')
  parser.add_argument('-v', '--verbose', action='count', default=0)
  parser.add_argument('--model', nargs='?', default='google/flan-t5-large')
  #parser.add_argument('--model', nargs='?', default='google/flan-t5-xl')
  parser.add_argument('--model-source', nargs='?', default='huggingface')

  args = parser.parse_args(args)

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

  # Load model
  if args.verbose > 0:
    print(f'Loading model {args.model} from {args.model_source}')
  model_load_begin_s = time.perf_counter()

  lm = ModelPack(model=args.model, source=args.model_source, model_args={
    'gpu': True,
    'do_sample': True,
  })

  model_load_end_s = time.perf_counter()
  if args.verbose > 0:
    print(f'{args.model} loaded in {model_load_end_s - model_load_begin_s:0.1f}s')

  parsed_data = dict()
  for t in table_readers:
    num_items = answer_questions(lm, doc_context, t.table_count_queries, parse_int)
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
          row_vals[field_name] = answer_questions(lm, doc_context, field_questions, t.row_field_parser_fn.get(field_name, parse_str) )

        parsed_data[t.table_name].append(row_vals)
      except:
        traceback.print_exc()

    if args.verbose > 0:
      print(f'parsed_data["{t.table_name}"] = {json.dumps(parsed_data[t.table_name], indent=3)}')











