"""Проверка эталонов и исходного кода D без исправлений: python verify_solutions.py."""
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits, load_iris, make_blobs, make_classification
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, root_mean_squared_error, silhouette_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

FOLDER = Path(__file__).resolve().parent

def references():
    result = {}
    def split(X, y, stratify=True):
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if stratify else None)
    Xtr, Xte, ytr, yte = split(*load_breast_cancer(return_X_y=True))
    model = GridSearchCV(make_pipeline(StandardScaler(), SVC()),
            {'svc__C': [0.1, 1, 10, 100], 'svc__gamma': ['scale', .001, .01, .1]}, cv=5).fit(Xtr, ytr)
    result['D01'] = [accuracy_score(yte, model.predict(Xte))]
    X, y = make_classification(n_samples=5000, weights=[.95, .05], random_state=42)
    Xtr, Xte, ytr, yte = split(X, y)
    pred = make_pipeline(StandardScaler(), LogisticRegression()).fit(Xtr, ytr).predict(Xte)
    result['D02'] = [f1_score(yte, pred, pos_label=1), balanced_accuracy_score(yte, pred)]
    Xtr, Xte, ytr, yte = split(*load_iris(return_X_y=True))
    model = GridSearchCV(make_pipeline(StandardScaler(), KNeighborsClassifier()),
                        {'kneighborsclassifier__n_neighbors': [1, 3, 5, 7, 9]}, cv=5).fit(Xtr, ytr)
    result['D03'] = [model.score(Xte, yte)]
    Xtr, Xte, ytr, yte = split(*load_digits(return_X_y=True))
    result['D04'] = [make_pipeline(StandardScaler(), PCA(n_components=.95), LogisticRegression(max_iter=2000)).fit(Xtr, ytr).score(Xte, yte),
                     make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr).score(Xte, yte)]
    X, _ = make_blobs(n_samples=500, centers=3, random_state=42)
    result['D05'] = [silhouette_score(X, KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)) for k in [2, 3, 4, 5]]
    Xtr, Xte, ytr, yte = split(*load_diabetes(return_X_y=True), stratify=False)
    model = GridSearchCV(make_pipeline(StandardScaler(), Ridge()), {'ridge__alpha': [.01, .1, 1, 10, 100]}, cv=5).fit(Xtr, ytr)
    result['D06'] = [root_mean_squared_error(yte, model.predict(Xte))]
    return result

def run():
    gold = references()
    (FOLDER / 'results' / 'reference_values.json').write_text(json.dumps(
        {'scikit-learn': version('scikit-learn'), 'seed': 42, 'D': gold}, ensure_ascii=False, indent=2) + '\n')
    checks = []
    env = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'LOKY_MAX_CPU_COUNT': '2'}
    for row in csv.DictReader((FOLDER / 'results/reviewer_1.csv').open(encoding='utf-8-sig')):
        if row['тип'] != 'D':
            continue
        path = FOLDER / 'results/review_code' / (row['case_id'] + '.py')
        started = time.perf_counter()
        try:
            process = subprocess.run([sys.executable, str(path)], capture_output=True, text=True,
                                     cwd=FOLDER, timeout=90, env=env)
            stdout, stderr, code = process.stdout, process.stderr, process.returncode
        except subprocess.TimeoutExpired:
            stdout, stderr, code = '', 'Лимит выполнения 90 секунд', 'timeout'
        numbers = [float(x) for x in re.findall(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', stdout)]
        correct = code == 0 and all(any(np.isclose(v, value, rtol=.01, atol=1e-10) for v in numbers) for value in gold[row['id']])
        note = ('Вывод содержит все требуемые метрики в допуске.' if correct else
                'Код не выполнился без исправлений.' if code != 0 else
                'Часть обязательного вывода отсутствует либо метрика вне допуска; вычисление без вывода не засчитывается как «выведите».')
        checks.append({'case_id': row['case_id'], 'id': row['id'], 'pass_at_1': int(code == 0),
                       'result_correct': int(correct), 'exit_code': code, 'stdout': stdout, 'stderr': stderr,
                       'seconds': round(time.perf_counter() - started, 3),
                       'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'note': note})
        print(row['id'], row['case_id'], 'pass', int(code == 0), 'result', int(correct), flush=True)
    (FOLDER / 'results/code_execution.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in checks))
    # Numeric and categorical golds are checked independently of any LLM answers.
    gini = lambda counts: 1 - sum((x / sum(counts)) ** 2 for x in counts)
    a = {'A01': .95*.01/(.95*.01+.05*.99), 'A02': gini([40,10]) - 37/50*gini([35,2]) - 13/50*gini([5,8]),
         'A03': 2/np.linalg.norm([3,4]), 'A04': (.4*.75*.75)/(.4*.75*.75+.6*(1/6)*(1/3)),
         'A05': np.mean((np.array([2,4,4])-np.array([1,5,3]))**2), 'A06': 1}
    bank = list(csv.DictReader((FOLDER / 'data/questions.csv').open(encoding='utf-8-sig')))
    gold_checks = [{'id': q['id'], 'verified': bool(np.isclose(a[q['id']], float(q['эталон_число'] or q['эталон_метка']), rtol=.01)), 'computed': a[q['id']]} for q in bank if q['тип']=='A']
    cancer = load_breast_cancer(as_frame=True)
    names = ['mean radius','mean texture','mean perimeter','mean area','mean concave points']
    e_values = [[17.99,10.38,122.8,1001,.1471],[13.61,24.98,88.05,582.7,.04489],[15.04,16.74,98.73,689.4,.06142]]
    for q, values in zip([q for q in bank if q['тип']=='E'], e_values):
        match = np.isclose(cancer.data[names], values, rtol=1e-6).all(axis=1)
        labels = cancer.target.loc[match]
        gold_checks.append({'id': q['id'], 'verified': len(labels)==1 and int(labels.iloc[0])==int(q['эталон_метка']),
                            'computed': int(labels.iloc[0]) if len(labels)==1 else None})
    (FOLDER / 'results/gold_checks.json').write_text(json.dumps(gold_checks, ensure_ascii=False, indent=2, default=float)+'\n')
    assert all(r['verified'] for r in gold_checks)
    print('Проверены эталоны A/E и заново вычислены эталоны D.', flush=True)

if __name__ == '__main__':
    run()
