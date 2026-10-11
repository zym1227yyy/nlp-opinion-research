"""Reproducible teaching simulation. No social-media posts are collected.
Run: python experiment.py --output results
Python 3.10+; standard library only. Seed 20261011.
"""
from pathlib import Path
from collections import Counter
import argparse, csv, hashlib, json, math, random, re, statistics

SEED = 20261011
COUNTS = [30, 40, 60, 100, 180, 240, 200, 160, 120, 80, 50, 40]
POS = ['方便','满意','支持','可靠','及时','合理','顺畅','周到','放心','透明','清楚']
NEG = ['糟糕','失望','混乱','麻烦','不满','焦虑','拥挤','敷衍','离谱','缓慢','不便']
NEGATORS = ['不','没','未','并非','并不','没有']
DEGREE = {'非常':1.5,'特别':1.5,'太':1.5,'有点':0.5,'稍微':0.5}
PHRASES = [
 ('出行更方便了',1,'direct'),('这次安排很合理',1,'direct'),('信息公开很透明',1,'direct'),
 ('大家可以放心',1,'direct'),('回应非常及时',1,'degree'),('服务特别周到',1,'degree'),
 ('整体让我满意',1,'direct'),('沟通比以前顺畅',1,'direct'),('我支持这个方案',1,'direct'),
 ('执行情况很可靠',1,'direct'),('情况说明很清楚',1,'direct'),
 ('结果并不糟糕',1,'negation'),('实际上没有不满',1,'negation'),('我并非失望',1,'negation'),
 ('现在没有麻烦',1,'negation'),('调整并不离谱',1,'negation'),
 ('安排实在糟糕',-1,'direct'),('这让我很失望',-1,'direct'),('现场特别混乱',-1,'degree'),
 ('给出行增加麻烦',-1,'direct'),('大家表示不满',-1,'direct'),('想到明天就焦虑',-1,'direct'),
 ('站台非常拥挤',-1,'degree'),('这个回应太敷衍',-1,'degree'),('安排有点离谱',-1,'degree'),
 ('处理速度缓慢',-1,'direct'),('现在出门很不便',-1,'direct'),
 ('我并不满意',-1,'negation'),('这种安排不合理',-1,'negation'),('我并非支持',-1,'negation'),
 ('这个解释不清楚',-1,'negation'),('现在并不方便',-1,'negation'),
 ('真是方便得让人天天迟到',-1,'irony'),('服务真周到连通知都没有',-1,'irony'),
 ('公告说明了调整时间',0,'neutral'),('明天会发布线路图',0,'neutral'),('我查到三项具体措施',0,'neutral'),
 ('请问后续是否还有说明',0,'neutral'),('方案需要观察几天',0,'neutral'),('有人正在整理问答',0,'neutral'),
 ('既有方便也有麻烦',0,'mixed'),('有支持也有不满',0,'mixed'),('既让人放心又让人焦虑',0,'mixed'),
 ('很满意但也很失望',0,'mixed'),('服务周到但处理缓慢',0,'mixed')]
TOPICS = ['公交线路调整','图书馆开放时间','社区停车方案','公园设施维护','校园食堂排队','公共办事预约']
PREFIX = ['刚看到消息','今天和朋友聊起','查阅公告后','从我的体验来看','关注了后续说明','大家正在讨论','补充一条观察','我想再问一句']
SUFFIX = ['希望继续公布进展','还需要听听其他人的看法','具体要看后续落实','准备再观察两天','建议补充详细说明','这个问题值得讨论','有新消息再补充','也想了解实际情况']
LEX = {x:1 for x in POS} | {x:-1 for x in NEG}
VOCAB = sorted(set(POS+NEG+NEGATORS+list(DEGREE)), key=lambda x:(-len(x),x))

def write_csv(p, rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def clean(t):
    t=re.sub(r'https?://\S+','',t)
    return re.sub(r'\s+','',t)

def tokens(t):
    # Longest-match dictionary for sentiment rules; retains punctuation boundaries.
    out=[];i=0
    while i<len(t):
        match=next((x for x in VOCAB if t.startswith(x,i)),None)
        if match:out.append(match);i+=len(match)
        else:out.append(t[i]);i+=1
    return out

def score(t, rules=True):
    ts=tokens(t);num=den=0.0
    for i,term in enumerate(ts):
        if term not in LEX:continue
        before=ts[max(0,i-3):i]
        # Punctuation closes scope; only three preceding tokens in same clause.
        for j in range(len(before)-1,-1,-1):
            if before[j] in '，。！？；': before=before[j+1:];break
        val=LEX[term];weight=1.0
        if rules:
            val*=(-1)**sum(x in NEGATORS for x in before)
            for x in before:weight*=DEGREE.get(x,1.0)
        num+=val*weight;den+=abs(weight)
    # +1 shrinks sparse evidence and preserves degree differences for one hit.
    return num/(den+1) if den else 0.0

def classify(s,tau=.05):return 1 if s>tau else -1 if s< -tau else 0

def features(t,n=2):
    spans=re.findall(r'[\u4e00-\u9fff]+',t)
    return Counter(s[i:i+n] for s in spans for i in range(max(0,len(s)-n+1)))

def vectors(rows,n=2,use_idf=True):
    f=[features(r['text_clean'],n) for r in rows];df=Counter()
    for v in f:df.update(v.keys())
    out=[]
    for v in f:
        vec={k:c*(math.log((1+len(f))/(1+df[k]))+1 if use_idf else 1) for k,c in v.items()}
        norm=math.sqrt(sum(z*z for z in vec.values()))
        out.append({k:z/norm for k,z in vec.items()} if norm else {})
    return out,df

def mean_cos(vecs):
    m=len(vecs)
    if m<2:return 0.0
    total=Counter()
    for v in vecs:total.update(v)
    return (sum(z*z for z in total.values())-sum(sum(z*z for z in v.values()) for v in vecs))/(m*(m-1))

def metrics(rows,key):
    cm={(a,b):0 for a in [-1,0,1] for b in [-1,0,1]}
    for r in rows:cm[(r['label_design'],r[key])]+=1
    fs=[]
    for c in [-1,0,1]:
        tp=cm[(c,c)];fp=sum(cm[(a,c)] for a in [-1,0,1] if a!=c);fn=sum(cm[(c,b)] for b in [-1,0,1] if b!=c)
        fs.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    return {'n':len(rows),'accuracy':sum(cm[(c,c)] for c in [-1,0,1])/len(rows),'macro_f1':sum(fs)/3,'confusion':[[cm[(a,b)] for b in [-1,0,1]] for a in [-1,0,1]]}

def entropy_weights(cols):
    d=[]
    for col in cols:
        lo=min(col);hi=max(col);z=[(x-lo)/(hi-lo) if hi!=lo else 0 for x in col]
        total=sum(z)
        e=-sum((x/total)*math.log(x/total) for x in z if x>0)/math.log(len(z)) if total else 1
        d.append(max(0,1-e))
    s=sum(d);return [x/s for x in d] if s else [1/len(d)]*len(d)

def run(out):
    out.mkdir(parents=True,exist_ok=True);rng=random.Random(SEED);rows=[]
    for day,n in enumerate(COUNTS,1):
        phase='萌芽' if day<=3 else '集中讨论' if day<=8 else '回落'
        probs=[.30,.35,.35] if day<=3 else [.65,.15,.20] if day<=8 else [.25,.40,.35]
        for j in range(n):
            label=rng.choices([-1,0,1],weights=probs)[0]
            candidates=[x for x in PHRASES if x[1]==label];phrase,_,kind=rng.choice(candidates)
            topic=TOPICS[0] if rng.random() < (.25 if day<=3 else .85 if day<=8 else .45) else rng.choice(TOPICS[1:])
            text=f'{rng.choice(PREFIX)}，关于{topic}，{phrase}。{rng.choice(SUFFIX)}。'
            rows.append({'id':f'S{len(rows)+1:05d}','day':day,'phase_design':phase,'topic_design':topic,'text':text,'label_design':label,'expression_type':kind,'source':'synthetic_template'})
    # Explicit faults: 40 exact repeats, 10 empties and 10 emoji-only rows.
    for r in rng.sample(rows,40):rows.append(dict(r,id=f'S{len(rows)+1:05d}',source='synthetic_duplicate'))
    for j in range(20):rows.append({'id':f'S{len(rows)+1:05d}','day':rng.randint(1,12),'phase_design':'噪声','topic_design':'无','text':'' if j<10 else '😊👍😊','label_design':0,'expression_type':'noise','source':'synthetic_noise'})
    raw=sorted(rows,key=lambda r:(r['day'],r['id']));write_csv(out/'data_raw.csv',raw)
    kept=[];seen=set();removed=[]
    for r in raw:
        t=clean(r['text']);reason='no_chinese' if not re.search(r'[\u4e00-\u9fff]',t) else 'duplicate' if t in seen else ''
        if reason:removed.append({'id':r['id'],'reason':reason});continue
        seen.add(t);s=score(t);base=score(t,False)
        kept.append(dict(r,text_clean=t,score_rule=s,pred_rule=classify(s),score_baseline=base,pred_baseline=classify(base)))
    write_csv(out/'data_processed.csv',kept);write_csv(out/'data_removed.csv',removed)
    vv,df=vectors(kept);daily=[]
    for day in range(1,13):
        ix=[i for i,r in enumerate(kept) if r['day']==day];rs=[kept[i] for i in ix]
        daily.append({'day':day,'n':len(rs),'similarity':mean_cos([vv[i] for i in ix]),'emotion_intensity':statistics.mean(abs(r['score_rule']) for r in rs),'negative_share':sum(r['pred_rule']==-1 for r in rs)/len(rs),'design_negative_share':sum(r['label_design']==-1 for r in rs)/len(rs)})
    cols=[[r[k] for r in daily] for k in ['n','similarity','emotion_intensity']];weights=entropy_weights(cols)
    zs=[[(x-min(c))/(max(c)-min(c)) if max(c)!=min(c) else 0 for x in c] for c in cols]
    for i,r in enumerate(daily):
        r['heat_equal']=100*sum(z[i] for z in zs)/3
        r['heat_entropy']=100*sum(w*z[i] for w,z in zip(weights,zs))
        r['volume_index']=100*zs[0][i]
    write_csv(out/'results_daily.csv',daily)
    stages=[]
    for phase in ['萌芽','集中讨论','回落']:
        ix=[i for i,r in enumerate(kept) if r['phase_design']==phase];rs=[kept[i] for i in ix]
        stages.append({'phase':phase,'n':len(rs),'similarity':mean_cos([vv[i] for i in ix]),'negative_share':sum(r['pred_rule']==-1 for r in rs)/len(rs),'design_negative_share':sum(r['label_design']==-1 for r in rs)/len(rs),'intensity':statistics.mean(abs(r['score_rule']) for r in rs)})
    write_csv(out/'results_stage.csv',stages)
    tests=[]
    for tau in [0,.05,.2,.5]:
        altered=[dict(r,pred_sensitivity=classify(r['score_rule'],tau)) for r in kept]
        m=metrics(altered,'pred_sensitivity');tests.append({'setting':f'threshold_{tau}','accuracy':m['accuracy'],'macro_f1':m['macro_f1']})
    for name,n,idf in [('bigram_tfidf',2,True),('bigram_tf',2,False),('trigram_tfidf',3,True)]:
        v,_=vectors(kept,n,idf)
        for p in ['萌芽','集中讨论','回落']:
            ix=[i for i,r in enumerate(kept) if r['phase_design']==p]
            tests.append({'setting':f'{name}_{p}','similarity':mean_cos([v[i] for i in ix])})
    write_csv(out/'results_sensitivity.csv',[{'setting':r['setting'],'accuracy':r.get('accuracy',''),'macro_f1':r.get('macro_f1',''),'similarity':r.get('similarity','')} for r in tests])
    types=[]
    for kind in sorted(set(r['expression_type'] for r in kept)):
        rs=[r for r in kept if r['expression_type']==kind]
        types.append({'expression_type':kind,'n':len(rs),'baseline_accuracy':metrics(rs,'pred_baseline')['accuracy'],'rule_accuracy':metrics(rs,'pred_rule')['accuracy']})
    write_csv(out/'results_by_expression.csv',types)
    errors=[{'id':r['id'],'text':r['text'],'design_label':r['label_design'],'pred_rule':r['pred_rule'],'expression_type':r['expression_type']} for r in kept if r['label_design']!=r['pred_rule']]
    if errors:write_csv(out/'error_examples.csv',errors)
    summary={'seed':SEED,'source':'entirely synthetic, fictional topics, no real users/events','raw_n':len(raw),'processed_n':len(kept),'removed':dict(Counter(r['reason'] for r in removed)),'retention':len(kept)/len(raw),'baseline':metrics(kept,'pred_baseline'),'rule':metrics(kept,'pred_rule'),'entropy_weights':weights,'stages':stages,'by_expression':types,'peak_day_equal':max(daily,key=lambda r:r['heat_equal'])['day'],'peak_day_entropy':max(daily,key=lambda r:r['heat_entropy'])['day'],'sensitivity':tests,'vocabulary_n':len(df),'char_count_mean':statistics.mean(len(r['text_clean']) for r in kept)}
    (out/'results_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    write_csv(out/'sentiment_lexicon.csv',[{'term':t,'polarity':v,'source':'teaching_authored'} for t,v in sorted(LEX.items())])
    resources={'negators':NEGATORS,'degree':DEGREE,'window_tokens':3,'threshold':.05,'phrases':[{'text':x[0],'label_design':x[1],'expression_type':x[2]} for x in PHRASES],'topics':TOPICS,'prefixes':PREFIX,'suffixes':SUFFIX,'daily_planned_counts':COUNTS}
    (out/'simulation_config.json').write_text(json.dumps(resources,ensure_ascii=False,indent=2),encoding='utf-8')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.suffix in ['.csv','.json'] and p.name!='checksums.json'}
    (out/'checksums.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    assert sum(r['n'] for r in daily)==len(kept)==sum(r['n'] for r in stages)
    assert all(0<=r['similarity']<=1 for r in daily)
    assert abs(sum(weights)-1)<1e-12
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output',default='results');args=a.parse_args();run(Path(args.output))
