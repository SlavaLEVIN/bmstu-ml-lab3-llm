"""Дополнительные E/HA/HB/HF-запросы. Запуск: python run_remaining_api.py gpt|gigachat."""
import concurrent.futures
import hashlib
import json
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
SYSTEM = 'Решай задачи по машинному обучению на русском. Отвечай кратко, строго в указанном формате.'
LOG = FOLDER / 'logs' / 'additional_api_responses.jsonl'
MAX_USD = 1.8

def run(name):
    keys = dict(line.split('=', 1) for line in (FOLDER / '.env').read_text().splitlines()
                if '=' in line and not line.lstrip().startswith('#'))
    keys = {k: v.strip().strip('\"\'') for k, v in keys.items()}
    context = ssl.create_default_context()
    context.load_verify_locations(str(FOLDER / 'certs/russian_trusted_root_ca.pem'))
    def post(url, data, headers):
        request = urllib.request.Request(url, data=data, headers=headers)
        return json.loads(urllib.request.urlopen(request, context=context, timeout=90).read())
    def authorize_gigachat():
        response = post('https://ngw.devices.sberbank.ru:9443/api/v2/oauth', b'scope=GIGACHAT_API_PERS',
                        {'Authorization': 'Basic ' + keys['GIGACHAT_CREDENTIALS'],
                         'Content-Type': 'application/x-www-form-urlencoded', 'RqUID': str(uuid.uuid4())})
        return response['access_token'], response['expires_at'] / 1000
    if name == 'gigachat':
        token, expires = authorize_gigachat()
        model_id, url = 'GigaChat-3-Ultra', 'https://api.giga.chat/v1/chat/completions'
    elif name == 'gpt':
        token = keys['LLM_GATE_API_KEY']
        model_id, url = 'gpt-6.1-sol', 'https://api.llm-gate.tech/v1/chat/completions'
    else:
        raise ValueError('Выбери gpt или gigachat')
    tasks = [json.loads(s) for filename in ['classification_prompts.jsonl', 'hypothesis_prompts.jsonl']
             for s in (FOLDER / 'data' / filename).read_text().splitlines()]
    records = [json.loads(s) for s in LOG.read_text().splitlines()] if LOG.exists() else []
    completed = {(r['model'], r['id'], r.get('prompt_hash')) for r in records
                 if r.get('answer') and not r.get('error') and r.get('finish_reason') != 'length'}
    lock = threading.Lock()
    spent = 0.20 + sum((r.get('prompt_tokens', 0) * 2 + r.get('completion_tokens', 0) * 10) / 1e6
                for r in records if r['model'] == 'gpt')
    reserved = 0
    stop = threading.Event()
    def one(task):
        nonlocal spent, reserved, token, expires
        prompt_hash = hashlib.sha256(task['prompt'].encode()).hexdigest()
        if (name, task['id'], prompt_hash) in completed or stop.is_set():
            return
        limit = 4096
        # Conservative byte-based input bound plus the maximum output, reserved before sending.
        bound = ((len((SYSTEM + task['prompt']).encode()) + 512) * 2 + limit * 10) / 1e6 if name == 'gpt' else 0
        while True:
            with lock:
                if name == 'gpt' and spent + bound > MAX_USD:
                    stop.set()
                    print('Остановлено: лимит дополнительного бюджета с резервом $1.80.', flush=True)
                    return
                if spent + reserved + bound <= MAX_USD or name != 'gpt':
                    reserved += bound
                    break
            time.sleep(0.2)
        payload = {'model': model_id, 'messages': [{'role': 'system', 'content': SYSTEM},
                   {'role': 'user', 'content': task['prompt']}]}
        payload.update({'reasoning_effort': 'low', 'max_completion_tokens': limit} if name == 'gpt'
                       else {'temperature': 0.0001, 'max_tokens': limit})
        record = {k: v for k, v in task.items() if k not in ['gold', 'prompt']}
        record.update(model=name, model_id=model_id, prompt_hash=prompt_hash,
                      date=datetime.now(timezone.utc).isoformat(), request=payload)
        started = time.perf_counter()
        try:
            if name == 'gigachat' and time.time() >= expires - 60:
                # GigaChat uses one worker; renew the 30-minute token during long runs.
                token, expires = authorize_gigachat()
            for attempt in range(3):
                record['attempts'] = attempt + 1
                try:
                    response = post(url, json.dumps(payload).encode(),
                                    {'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
                    break
                except urllib.error.HTTPError as error:
                    if error.code not in [429, 500, 502, 503, 504] or attempt == 2:
                        raise
                    time.sleep(min(30, 2 ** (attempt + 1)))
            choice, usage = response['choices'][0], response.get('usage', {})
            answer = choice['message'].get('content') or ''
            record.update(answer=answer, finish_reason=choice.get('finish_reason'),
                          prompt_tokens=usage.get('prompt_tokens', 0),
                          completion_tokens=usage.get('completion_tokens', 0), tokens=usage.get('total_tokens'))
            if not answer or record['finish_reason'] == 'length':
                raise ValueError('Нет полного ответа')
            if task['stage'] == 'classification':
                record['prediction'] = int(answer.strip()) if answer.strip() in ['0', '1'] else -1
        except Exception as error:
            record['error'] = str(error)[:250]
            if isinstance(error, urllib.error.HTTPError) and error.code in [401, 402, 403]:
                stop.set()
        finally:
            record['seconds'] = round(time.perf_counter() - started, 3)
            with lock:
                reserved -= bound
                if name == 'gpt':
                    spent += (record.get('prompt_tokens', 0) * 2 + record.get('completion_tokens', 0) * 10) / 1e6
                with LOG.open('a', encoding='utf-8') as file:
                    file.write(json.dumps(record, ensure_ascii=False) + '\n')
                print(name, task['id'], record.get('error', 'готово'), f'USD≤{spent:.4f}', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=12 if name == 'gpt' else 1) as pool:
        list(pool.map(one, tasks))
    print('Прогон завершён. Журнал:', LOG, flush=True)

if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else 'gpt')
