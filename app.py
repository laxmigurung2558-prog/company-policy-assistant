import csv, os, re, math, time, json, urllib.request, urllib.error
from collections import Counter
from flask import Flask, jsonify, render_template, request
from pathlib import Path

BASE = Path(__file__).resolve().parent
app = Flask(__name__)
with open(BASE / 'company_policies.csv', newline='', encoding='utf-8-sig') as f:
    POLICIES = list(csv.DictReader(f))

STOP = set('a an the is are was were do does did can could should would may i we you they he she it my our their about for from to of on in at by with and or if how what when where which who much many long company employees employee policy policies rules rule tell me please'.split())
def tokens(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP]

def label(p):
    return {'title':p['title'], 'department':p['department'],'category':p['category'], 'policy_text':p['policy_text']}

DOCS = [tokens(' '.join([p['title'],p['title'],p['category'],p['department'],p['policy_text']])) for p in POLICIES]
DF = Counter(t for doc in DOCS for t in set(doc))
IDF = {t: math.log((len(DOCS)+1)/(df+1))+1 for t,df in DF.items()}
def vector(ts):
    counts=Counter(ts)
    v={t: (1+math.log(n))*IDF.get(t, math.log(len(DOCS)+1)) for t,n in counts.items()}
    norm=math.sqrt(sum(x*x for x in v.values())) or 1
    return {t:x/norm for t,x in v.items()}
VECTORS=[vector(d) for d in DOCS]

def rules_search(q):
    terms=set(tokens(q))
    ranked=[]
    for p in POLICIES:
        title=set(tokens(p['title'])); body=set(tokens(p['policy_text'])); other=set(tokens(p['category']+' '+p['department']))
        score=4*len(terms&title)+len(terms&body)+0.5*len(terms&other)
        if score: ranked.append((score,p))
    ranked.sort(key=lambda x:-x[0])
    return [p for _,p in ranked[:5]]

def semantic_search(q):
    query=vector(tokens(q))
    scores=[sum(value*v.get(term,0) for term,value in query.items()) for v in VECTORS]
    ordered=sorted(range(len(scores)),key=lambda i:scores[i],reverse=True)
    return [POLICIES[i] for i in ordered[:5] if scores[i]>0]

SYSTEM = '''You are a careful company policy assistant. Only answer using the supplied company policies. Never invent requirements, amounts, dates, exceptions or approvals. If there is no policy that answers the question, say clearly that the available policies do not specify it. Respond ONLY with valid JSON, no markdown: {"answer":"...", "policy_titles":["Exact Policy Title"],"unsupported":false}. Set unsupported=true if the policies do not support a substantive answer. Reference exact title(s) supplied. Be concise.'''

def ask_gemini(question, rows):
    key=os.environ.get('GEMINI_API_KEY','').strip()
    if not key: raise RuntimeError('Gemini is not configured. Add GEMINI_API_KEY to the deployment environment.')
    model=os.environ.get('GEMINI_MODEL','gemini-2.5-flash')
    records=[label(p) for p in rows]
    prompt='QUESTION: '+question+'\nAVAILABLE POLICIES:\n'+json.dumps(records,ensure_ascii=False)
    url=f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}'
    payload={'systemInstruction': {'parts':[{'text':SYSTEM}]}, 'contents':[{'role':'user','parts':[{'text':prompt}]}], 'generationConfig':{'temperature':0,'responseMimeType':'application/json'}}
    req=urllib.request.Request(url,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'},method='POST')
    try:
        with urllib.request.urlopen(req,timeout=35) as response: data=json.load(response)
    except urllib.error.HTTPError as exc:
        detail=exc.read().decode(errors='ignore')[:400]
        if exc.code==429: raise RuntimeError('Gemini rate limit reached (429). Please retry later or check your API quota.')
        raise RuntimeError(f'Gemini returned HTTP {exc.code}: {detail}')
    raw=''.join(part.get('text','') for part in data['candidates'][0]['content']['parts'])
    result=json.loads(raw.strip().removeprefix('```json').removesuffix('```').strip())
    usage=data.get('usageMetadata',{})
    valid={p['title']:p for p in rows}
    titles=[t for t in result.get('policy_titles',[]) if t in valid]
    return {'answer':str(result.get('answer','No answer provided.')),'policies':[label(valid[t]) for t in titles], 'tokens':usage.get('totalTokenCount'), 'unsupported':bool(result.get('unsupported',False)), 'support_note':'AI-reported support status; not an independent factual verification.'}

@app.get('/')
def home():
    return render_template('index.html', count=len(POLICIES), categories=len(set(p['category'] for p in POLICIES)), configured=bool(os.environ.get('GEMINI_API_KEY')))

@app.post('/api/compare')
def compare():
    body=request.get_json(silent=True) or {}
    question=str(body.get('question','')).strip()
    if not 3<=len(question)<=500: return jsonify({'error':'Enter a question of 3–500 characters.'}),400
    start=time.perf_counter(); found=rules_search(question); rule_ms=round((time.perf_counter()-start)*1000,2)
    if found:
        best=found[0]
        answer=f"According to the {best['title']}: {best['policy_text']}"
    else: answer='No matching policy was found. The policy database may not cover this question.'
    data={'rules':{'answer':answer,'policies':[label(p) for p in found[:3]],'ms':rule_ms,'tokens':0,'unsupported':not bool(found),'support_note':'Deterministic extract from the policy database; match relevance is not guaranteed.'}}
    if not os.environ.get('GEMINI_API_KEY'):
        msg='Not run: Set GEMINI_API_KEY to enable a live model comparison.'
        data['plain']={'error':msg};data['rag']={'error':msg}
        return jsonify(data)
    for name,rows in [('plain',POLICIES),('rag',semantic_search(question))]:
        start=time.perf_counter()
        try:
            if name=='rag' and not rows:
                result={'answer':'No matching policy found in the vector index.', 'policies':[], 'tokens':0,'unsupported':True,'support_note':'No vector match; model was not called.'}
            else:result=ask_gemini(question,rows)
            result['ms']=round((time.perf_counter()-start)*1000,2)
            data[name]=result
        except Exception as exc:
            data[name]={'error':str(exc),'ms':round((time.perf_counter()-start)*1000,2)}
    return jsonify(data)

@app.get('/api/policies')
def policies():
    query=request.args.get('q','').lower().strip()
    rows=[label(p) for p in POLICIES if query in ' '.join(p.values()).lower()]
    return jsonify({'policies':rows[:100], 'count':len(rows)})

if __name__=='__main__':
    app.run(host='0.0.0.0',port=int(os.getenv('PORT','5000')))
