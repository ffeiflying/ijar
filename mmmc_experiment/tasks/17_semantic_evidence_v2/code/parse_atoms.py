"""Question-only semantic atoms with token/character alignment. No labels/answers."""
import difflib
import re

GENERIC={'color','colour','kind','type','number','image','picture','photo','photograph','thing','one','something','anything'}
RELATIONS={'left','right','above','below','behind','under','over','inside','outside','beside','near','between','on','in','next','front','at','with','without','through','across','against','around','along','before','after','toward','towards'}
SKIP_VERBS={'be','do','have','see','show','depict','take','look','appear','seem'}
SYNONYMS={'bicycle':['bike'],'bike':['bicycle'],'sofa':['couch'],'couch':['sofa'],'car':['automobile'],'automobile':['car'],'airplane':['aeroplane','plane'],'aeroplane':['airplane','plane'],'television':['tv'],'tv':['television'],'gray':['grey'],'grey':['gray'],'motorcycle':['motorbike'],'motorbike':['motorcycle']}

def parse(question,tokenizer,nlp):
    ids=tokenizer.encode(question,add_special_tokens=False)
    decoded=tokenizer.decode(ids,skip_special_tokens=True)
    charmap={}
    for block in difflib.SequenceMatcher(None,decoded,question,autojunk=False).get_matching_blocks():
        for j in range(block.size):
            charmap[block.a+j]=block.b+j
    ends=[0]+[len(tokenizer.decode(ids[:i+1],skip_special_tokens=True)) for i in range(len(ids))]
    ranges=[]
    for a,b in zip(ends[:-1],ends[1:]):
        chars=[charmap[i] for i in range(a,b) if i in charmap and not decoded[i].isspace()]
        ranges.append(set(chars))
    doc=nlp(question)
    atoms=[]
    for token in doc:
        surface=token.text.lower()
        lemma=token.lemma_.lower()
        if not re.search('[a-z0-9]',surface):
            continue
        if token.pos_ in ['NOUN','PROPN'] and lemma not in GENERIC:
            kind='object'
        elif surface in RELATIONS or (token.pos_=='VERB' and lemma not in SKIP_VERBS):
            kind='relation'
        elif token.pos_=='ADJ' and surface not in ['many','much','other','else','several','few']:
            kind='attribute'
        elif token.pos_=='NUM':
            kind='count'
        else:
            continue
        span=set(range(token.idx,token.idx+len(token.text)))
        qindices=[i for i,s in enumerate(ranges) if s&span]
        if not qindices:
            continue
        names=set([surface,lemma]+SYNONYMS.get(lemma,[]))
        aliases=set()
        for name in sorted(names):
            for value in [name,' '+name,name.capitalize()]:
                encoded=tokenizer.encode(value,add_special_tokens=False)
                if len(encoded)==1:
                    aliases.add(encoded[0])
        atoms.append({'surface':token.text,'lemma':lemma,'kind':kind,'start':token.idx,'end':token.idx+len(token.text),'qindices':qindices,'aliases':sorted(aliases),'head_start':token.head.idx,'negated':any(c.dep_=='neg' for c in token.children) or surface=='without'})
    if not atoms:
        indices=[i for i,v in enumerate(ids) if re.search('[a-zA-Z0-9]',tokenizer.decode([v]))]
        atoms=[{'surface':question,'lemma':'global','kind':'global','start':0,'end':len(question),'qindices':indices if indices else list(range(len(ids))),'aliases':[],'head_start':0,'negated':False}]
    for i,atom in enumerate(atoms):
        atom['anchor']=-1
        atom['subject']=-1
        atom['object']=-1
        if atom['kind']=='attribute':
            options=[j for j,a in enumerate(atoms) if a['kind']=='object' and a['start']==atom['head_start']]
            if options:
                atom['anchor']=options[0]
        if atom['kind']=='relation':
            left=[j for j,a in enumerate(atoms) if a['kind']=='object' and a['end']<=atom['start']]
            right=[j for j,a in enumerate(atoms) if a['kind']=='object' and a['start']>=atom['end']]
            if left:
                atom['subject']=left[-1]
            if right:
                atom['object']=right[0]
    return {'question':question,'qids':ids,'decoded_question':decoded,'atoms':atoms,'alignment_fraction':len(charmap)/max(1,len(question))}
