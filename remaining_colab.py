"""Запускается внутри Colab через %run -i после загрузки colab_inputs.zip."""
import csv
import gc
import hashlib
import json
import time
import zipfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import torch
from google.colab import drive, files
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

DATA = Path('/content/lab_remaining')
FOLDER = Path('/content/drive/MyDrive/Lab3_4')
drive.mount('/content/drive')
FOLDER.mkdir(exist_ok=True)
settings = json.loads((DATA / 'generation_settings.json').read_text())
SYSTEM, FORMATS = settings['system'], settings['formats']
extra = [json.loads(s) for filename in ['classification_prompts.jsonl', 'hypothesis_prompts.jsonl']
         for s in (DATA / filename).read_text().splitlines()]
questions = list(csv.DictReader((DATA / 'questions.csv').open(encoding='utf-8-sig')))
bank = [{'id': q['id'], 'тип': q['тип'], 'stage': 'bank',
         'prompt': FORMATS[q['тип']] + '\n\n' + q['текст_ru']} for q in questions]
LOG = FOLDER / 'colab_experiments.jsonl'
records = [json.loads(s) for s in LOG.read_text().splitlines()] if LOG.exists() else []

def run_set(name, model_obj, tokenizer_obj, tasks, quantization):
    completed = {(r['model'], r['id'], r['prompt_hash']) for r in records
                 if r.get('answer') and not r.get('error') and r.get('finish_reason') != 'length'}
    env = {'model': name, 'model_id': model_obj.config._name_or_path,
           'gpu': torch.cuda.get_device_name(0), 'revision': model_obj.config._commit_hash,
           'quantization': quantization, 'seed': 42,
           'libraries': {p: version(p) for p in ['torch', 'transformers', 'accelerate']}}
    limits = {'A': 768, 'B': 512, 'C': 768, 'D': 2048, 'E': 32, 'F': 768}
    for index, task in enumerate(tasks, 1):
        fingerprint = hashlib.sha256(task['prompt'].encode()).hexdigest()
        if (name, task['id'], fingerprint) in completed:
            continue
        messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': task['prompt']}]
        record = {k: v for k, v in task.items() if k not in ['gold', 'prompt']}
        record.update(**env, date=datetime.now(timezone.utc).isoformat(), prompt_hash=fingerprint,
                      request={'model': env['model_id'], 'messages': messages,
                               'do_sample': False, 'max_new_tokens': limits[task['тип']]})
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            inputs = tokenizer_obj.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                       return_tensors='pt', return_dict=True).to(model_obj.device)
            with torch.inference_mode():
                output = model_obj.generate(**inputs, do_sample=False, max_new_tokens=limits[task['тип']],
                                            pad_token_id=tokenizer_obj.pad_token_id)
            torch.cuda.synchronize()
            generated = output[0, inputs['input_ids'].shape[1]:]
            answer = tokenizer_obj.decode(generated, skip_special_tokens=True).strip()
            eos = model_obj.generation_config.eos_token_id
            eos = [eos] if isinstance(eos, int) else eos
            complete = len(generated) and generated[-1].item() in eos
            record.update(answer=answer, prompt_tokens=inputs['input_ids'].shape[1],
                          completion_tokens=len(generated), tokens=output.shape[1],
                          finish_reason='stop' if complete else 'length')
            if not complete or not answer:
                raise ValueError('Ответ не завершён; увеличь лимит и повтори только пропуски.')
            if task['stage'] == 'classification':
                record['prediction'] = int(answer) if answer in ['0', '1'] else -1
        except (Exception, KeyboardInterrupt) as error:
            record['error'] = str(error)[:300] or 'Выполнение прервано'
            if isinstance(error, (KeyboardInterrupt, torch.OutOfMemoryError)):
                raise
        finally:
            record.update(seconds=round(time.perf_counter() - started, 3),
                          peak_gpu_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3))
            with LOG.open('a', encoding='utf-8') as file:
                file.write(json.dumps(record, ensure_ascii=False) + '\n')
            records.append(record)
        if index % 20 == 0 or task['stage'] != 'classification' or record.get('error'):
            print(name, f'{index}/{len(tasks)}', task['id'], record.get('error', 'готово'), flush=True)

if not torch.cuda.is_available():
    raise RuntimeError('Нужна среда Colab с GPU T4.')
set_seed(42)
try:
    if 'model' not in globals() or model.config._name_or_path != 'Qwen/Qwen3-4B-Instruct-2507':
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Instruct-2507')
        model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-4B-Instruct-2507', torch_dtype=torch.float16,
                    device_map={'': 'cuda:0'}, attn_implementation='sdpa', low_cpu_mem_usage=True).eval()
    run_set('local', model, tokenizer, extra, 'none; float16')
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained('t-tech/T-lite-it-2.1')
    model = AutoModelForCausalLM.from_pretrained('t-tech/T-lite-it-2.1', device_map={'': 'cuda:0'},
                torch_dtype=torch.float16, attn_implementation='sdpa', low_cpu_mem_usage=True,
                quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)).eval()
    run_set('tlite', model, tokenizer, bank + extra, 'NF4 4-bit double quantization; float16 compute')
finally:
    archive = Path('/content/colab_results.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        if LOG.exists():
            bundle.write(LOG, 'colab_experiments.jsonl')
    print('Журнал сохранён:', LOG, flush=True)
    files.download(str(archive))
