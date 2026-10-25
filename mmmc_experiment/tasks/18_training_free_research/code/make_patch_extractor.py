"""Typed semantic frame readout on visual patches, single frozen-model prefill."""
from pathlib import Path
TASK=Path(__file__).resolve().parent.parent
source=(TASK.parent/'17_semantic_evidence_v2/code/extract.py').read_text()
source=source.replace('from parse_atoms import parse',"sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'17_semantic_evidence_v2/code'))\nfrom parse_atoms import parse")
# Reuse fixed dictionary construction only; no source extraction or model fitting.
generator=(TASK/'code/make_source_extractor.py').read_text()
import ast
module=ast.parse(generator)
dictionary=next(ast.literal_eval(n.value) for n in module.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='dictionary' for t in n.targets))
source=source.replace("    nlp=spacy.load('en_core_web_sm',disable=['ner'])",dictionary+"    nlp=spacy.load('en_core_web_sm',disable=['ner'])")
needle="            data[pfx+'atom_head_logp']=atom_head.cpu().numpy()"
inject='''            frames=[]
            for atom,ix in zip(state['meta']['atoms'],state['atom_positions']):
                ids=sorted(set(atom['aliases'] if atom['aliases'] else [state['qids'][j] for j in ix]))
                groups=[ids];owned=set(ids)
                for g in semantic_groups.get(atom['kind'],semantic_groups['object']):
                    if set(g)&set(ids):continue
                    unique=sorted(set(g)-owned)
                    if unique:groups.append(unique);owned.update(unique)
                frames.append(groups)
            na=len(frames);nk=max(map(len,frames));nm=max(len(g) for frame in frames for g in frame)
            ci=np.zeros((na,nk,nm),np.int64);cm=np.zeros((na,nk,nm),bool)
            for i,frame in enumerate(frames):
                for j,g in enumerate(frame):ci[i,j,:len(g)]=g;cm[i,j,:len(g)]=True
            index=torch.from_numpy(ci).to(q.device);valid=torch.from_numpy(cm).to(q.device)
            # Shape[A,K,P]: label events within the same local-description frame.
            selected=full[:,index].masked_fill(~valid[None],float('-inf'))
            patch_frames=(selected.logsumexp(-1)-norm[:,None,None]).permute(1,2,0)
            head_frames=(patch_frames[None,:,:,:]+atom_conditional[:,:,None,:].clamp_min(1e-30).log()).logsumexp(-1)
            data[pfx+'semantic_frame_mask']=cm.any(-1)
            data[pfx+'semantic_frame_token_ids']=ci
            data[pfx+'semantic_patch_logp']=patch_frames.nan_to_num(neginf=-1e30).cpu().numpy()
            data[pfx+'semantic_head_logp']=head_frames.nan_to_num(neginf=-1e30).cpu().numpy()
'''
assert needle in source;source=source.replace(needle,inject+needle)
source=source.replace("'version':1,'layers':layers","'version':4,'semantic_dictionary':semantic_words,'semantic_frame':'competing descriptions of queried local entity, not mutually exclusive whole-image objects','layers':layers")
source=source.replace('results/features/{args.split}','results/patch_v4/{args.split}')
source=source.replace('results/feature_config.json','results/patch_v4_config.json').replace('results/feature_config_{args.split}.json','results/patch_v4_config_{args.split}.json')
(TASK/'code/extract_patch_frames.py').write_text(source)
