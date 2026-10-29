"""Сводка дополнительных экспериментов: python analyze_experiments.py."""
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FOLDER = Path(__file__).resolve().parent
OUT = FOLDER / 'results'

def metric(y, pred):
    rare = y == 0
    tp = np.sum(rare & (pred == 0), axis=-1)
    fp = np.sum(~rare & (pred == 0), axis=-1)
    fn = np.sum(rare & (pred != 0), axis=-1)
    f1 = np.divide(2*tp, 2*tp+fp+fn, out=np.zeros_like(tp, dtype=float), where=(2*tp+fp+fn)!=0)
    ba = .5*(np.sum(rare & (pred==0), axis=-1)/np.sum(rare, axis=-1)
              + np.sum(~rare & (pred==1), axis=-1)/np.sum(~rare, axis=-1))
    return f1, ba

def number(text):
    tail = re.split('ответ\\s*:', text, flags=re.I)[-1].replace('{,}', '.')
    tail = re.sub(r'\\(?:d?frac)\s*\{([-+\d.,]+)\}\s*\{([-+\d.,]+)\}', r'\1/\2', tail)
    found = re.findall(r'[-+]?\d+(?:[.,]\d+)?(?:\s*/\s*[-+]?\d+(?:[.,]\d+)?)?\s*%?', tail)
    if not found: return np.nan
    token = found[-1].strip(); percent='%' in token
    parts=[float(x) for x in token.replace('%','').replace(',','.').split('/')]
    value=parts[0]/parts[1] if len(parts)==2 and parts[1] else parts[0]
    return value/100 if percent else value

def run():
    records = [json.loads(line) for filename in ['additional_api_responses.jsonl', 'colab_experiments.jsonl']
               if (path := FOLDER / 'logs' / filename).exists() for line in path.read_text().splitlines()]
    selected = {}
    for r in records:
        key=(r['model'],r['id'])
        if key not in selected and r.get('answer') and not r.get('error') and r.get('finish_reason')!='length':
            selected[key]=r
    predictions=pd.DataFrame([{'model':r['model'],'shots':r['shots'],'row_id':int(r['row_id']),
                              'prediction':r.get('prediction',int(r['answer'].strip()) if r['answer'].strip() in ['0','1'] else -1)}
                             for r in selected.values() if r.get('stage')=='classification'],
                             columns=['model','shots','row_id','prediction'])
    predictions.to_csv(FOLDER/'data/classification_predictions.csv',index=False)
    test=pd.read_csv(FOLDER/'data/classification_test.csv')
    support=pd.read_csv(FOLDER/'data/classification_support.csv')
    features=support.columns.drop(['row_id','target','in_5_shot'])
    y=test.target.to_numpy()
    boot=np.random.default_rng(42).integers(0,len(test),(1000,len(test)))
    baseline={}
    for shots in [5,20]:
        train=support.loc[support.in_5_shot.eq(1)] if shots==5 else support
        for name,model in [('LogReg',make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,random_state=42))),('GaussianNB',GaussianNB())]:
            baseline[(name,shots)]=model.fit(train[features],train.target).predict(test[features])
    baseline[('always_majority',20)]=np.full(len(test),support.target.mode().iloc[0])
    all_predictions=dict(baseline)
    coverage=[]
    for (name,shots),group in predictions.groupby(['model','shots']):
        coverage.append({'model':name,'shots':shots,'answered':len(group),'planned':150})
        if len(group)==150 and not group.row_id.duplicated().any() and set(group.row_id)==set(test.row_id):
            all_predictions[(name,shots)]=test[['row_id']].merge(group,on='row_id').prediction.to_numpy()
    metrics=[]; comparisons=[]
    for (name,shots),pred in all_predictions.items():
        f1,ba=metric(y,pred); sample_f1,sample_ba=metric(y[boot],pred[boot])
        fci=np.quantile(sample_f1,[.025,.975]);bci=np.quantile(sample_ba,[.025,.975])
        metrics.append({'model':name,'shots':shots,'n':150,'F1_rare':f1,'F1_low':fci[0],'F1_high':fci[1],
                        'balanced_accuracy':ba,'BA_low':bci[0],'BA_high':bci[1],'accuracy':np.mean(y==pred),
                        'invalid_labels':np.sum(~np.isin(pred,[0,1]))})
        if name not in ['LogReg','GaussianNB','always_majority'] and shots in [5,20]:
            ref=baseline[('LogReg',shots)]
            rf,rb=metric(y,ref);bsf,bsb=metric(y[boot],ref[boot])
            fdelta=np.quantile(sample_f1-bsf,[.025,.975]);bdelta=np.quantile(sample_ba-bsb,[.025,.975])
            comparisons.append({'model':name,'shots':shots,'F1_delta_LogReg':f1-rf,'F1_delta_low':fdelta[0],
                                'F1_delta_high':fdelta[1],'BA_delta_LogReg':ba-rb,'BA_delta_low':bdelta[0],'BA_delta_high':bdelta[1]})
    metrics=pd.DataFrame(metrics);metrics.to_csv(OUT/'classification_with_ci.csv',index=False)
    pd.DataFrame(comparisons).to_csv(OUT/'hypothesis_H12.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'classification_coverage.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    for ax,column in zip(axes,['F1_rare','balanced_accuracy']):
        for name,group in metrics.groupby('model'):
            if name=='always_majority':
                ax.axhline(group[column].iloc[0],color='gray',linestyle=':',label=name)
            else:ax.plot(group.shots,group[column],marker='o',label=name)
        ax.set(xlabel='Число примеров',ylabel=column,xticks=[0,5,20],ylim=(0,1.05))
    axes[1].legend(fontsize=8);fig.savefig(OUT/'classification_few_shot.png',dpi=150);plt.close(fig)
    tasks={r['id']:r for r in map(json.loads,(FOLDER/'data/hypothesis_prompts.jsonl').read_text().splitlines())}
    checked=[]
    for r in selected.values():
        if r.get('stage')!='hypotheses':continue
        task=tasks[r['id']];kind=task['тип'];answer=r['answer']
        if kind=='A':correct=int(np.isclose(number(answer),float(task['gold']),rtol=.01))
        elif kind=='B':
            match=re.search(r'\b(неверно|верно)\b',answer,re.I)
            correct=int(bool(match) and match[0].lower()==task['gold'])
        else:
            correct=int(bool(re.search(r'нельзя|невозмож|недостат|не определ|неизвест|без.{0,50}(значени|дол|сведени)|не указ|не дан',answer,re.I)))
        checked.append({'model':r['model'],'id':r['id'],'тип':kind,'correct':correct,
                        'assessment_source':'automatic' if kind!='F' else 'heuristic_pending_human','answer':answer})
    h=pd.DataFrame(checked,columns=['model','id','тип','correct','assessment_source','answer'])
    flat=h.copy(); flat['answer']=flat.answer.str.replace('\n',r'\n',regex=False)
    flat.to_csv(OUT/'hypothesis_answer_checks.csv',index=False)
    by_type=h.groupby(['model','тип']).correct.agg(['mean','count']).reset_index()
    by_type.to_csv(OUT/'hypothesis_scores.csv',index=False)
    h3=by_type.loc[by_type['тип'].eq('F')].rename(columns={'mean':'correct_insufficiency_rate','count':'n'})
    h3.to_csv(OUT/'hypothesis_H3.csv',index=False)
    h5=[]
    pair=h.loc[h['тип'].isin(['A','B'])].pivot(index='id',columns='model',values='correct')
    if all(name in pair for name in ['gigachat','gpt','local']):
        types={qid:tasks[qid]['тип'] for qid in pair.index}
        a=pair.loc[[qid for qid in pair.index if types[qid]=='A'],['gigachat','gpt','local']].dropna()
        b=pair.loc[[qid for qid in pair.index if types[qid]=='B'],['gigachat','gpt','local']].dropna()
        if len(a)==len(b)==6:
            gap_a=a[['gigachat','gpt']].mean(axis=1).to_numpy()-a.local.to_numpy()
            gap_b=b[['gigachat','gpt']].mean(axis=1).to_numpy()-b.local.to_numpy()
            rng=np.random.default_rng(42)
            delta=rng.choice(gap_a,(1000,len(a))).mean(axis=1)-rng.choice(gap_b,(1000,len(b))).mean(axis=1)
            ci=np.quantile(delta,[.025,.975])
            h5=[{'gap_A':gap_a.mean(),'gap_B':gap_b.mean(),'difference':gap_a.mean()-gap_b.mean(),'ci_low':ci[0],'ci_high':ci[1],'n_A':len(a),'n_B':len(b)}]
    pd.DataFrame(h5,columns=['gap_A','gap_B','difference','ci_low','ci_high','n_A','n_B']).to_csv(OUT/'hypothesis_H5.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,3),layout='constrained')
    if len(by_type):
        chart=by_type.pivot(index='model',columns='тип',values='mean')
        chart.plot.bar(ax=ax);ax.set(ylabel='Accuracy; F — предварительная разметка',ylim=(0,1.1),title='Новые контрольные задачи H3/H5');ax.tick_params(axis='x',rotation=0)
    fig.savefig(OUT/'hypothesis_controls.png',dpi=150);plt.close(fig)
    print('Полные группы классификации:',[(k,len(v)) for k,v in all_predictions.items() if k not in baseline])
    print('Ответов дополнительных задач:',len(h))

if __name__=='__main__':run()
