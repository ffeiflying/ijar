"""Build isolated extractor from the preserved feature extractor with passive source reads."""
from pathlib import Path
TASK=Path(__file__).resolve().parent.parent
source=(TASK.parent/'17_semantic_evidence_v2/code/extract.py').read_text()
source=source.replace('from parse_atoms import parse',"sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'17_semantic_evidence_v2/code'))\nfrom parse_atoms import parse")
dictionary="""    semantic_words={'object':'person man woman boy girl dog cat horse sheep cow bird elephant bear zebra giraffe bicycle car motorcycle airplane bus train truck boat chair couch bed bench table toilet television laptop mouse keyboard phone microwave oven toaster sink refrigerator book clock vase scissors umbrella bag suitcase bottle cup fork knife spoon bowl banana apple orange sandwich broccoli carrot pizza donut cake ball bat glove skateboard surfboard racket kite sign light hydrant parking meter backpack tie handbag skis snowboard tennis baseball street tree grass building water sky road beach ocean mountain snow field kitchen room plate food shirt hat shoe animal vehicle furniture plant flower fruit bread fish chicken duck boat fence window door roof umbrella camera bicycle'.split(),'attribute':'red blue green yellow orange purple pink brown black white gray silver golden large small tall short long round square striped spotted wooden metal plastic empty full open closed wet dry old young'.split(),'relation':'left right above below behind front inside outside under over near beside on in next holding wearing riding sitting standing walking running flying eating drinking driving'.split(),'count':['one','two','three','four','five','six','seven','eight','nine','ten']}
    semantic_groups={}
    for kind,words in semantic_words.items():
        semantic_groups[kind]=[]
        for word in sorted(set(words)):
            ids=set()
            for spelling in [word,' '+word,word.capitalize()]:
                token_ids=tokenizer.encode(spelling,add_special_tokens=False)
                if len(token_ids)==1:ids.add(token_ids[0])
            if ids:semantic_groups[kind].append(sorted(ids))
"""
source=source.replace("    nlp=spacy.load('en_core_web_sm',disable=['ner'])",dictionary+"    nlp=spacy.load('en_core_web_sm',disable=['ner'])")
source=source.replace("weights=logits.softmax(-1)[0,:,:,visual]", "all_weights=logits.softmax(-1)[0]\n            weights=all_weights[:,:,visual]")
needle="            conditional=weights/weights.sum(-1,keepdim=True).clamp_min(1e-30)"
inject='''            # Passive decomposition of this actual attention operation; no rerun.
            values=helper.repeat_kv(v,attn.num_key_value_groups)[0].float()
            image_part=all_weights[:,:,visual]@values[:,visual]
            other_part=all_weights[:,:,~visual]@values[:,~visual]
            def project(part):
                part=part.transpose(0,1).reshape(len(qpos),-1)
                return torch.nn.functional.linear(part,attn.wo.weight.float(),None)
            ov=project(image_part);ot=project(other_part)
            bias=attn.wo.bias
            if bias is not None:ot=ot+bias.float()
            base=all_hidden[qpos].float()-ov-ot
            source_states={'visual':ov,'text':ot,'visual_residual':base+ov,'text_residual':base+ot,'base':base,'joint':all_hidden[qpos].float()}
            frames=[]
            for atom,ix in zip(state['meta']['atoms'],state['atom_positions']):
                ids=sorted(set(atom['aliases'] if atom['aliases'] else [state['qids'][j] for j in ix]))
                groups=[ids];owned=set(ids)
                for g in semantic_groups.get(atom['kind'],semantic_groups['object']):
                    if set(g)&set(ids):continue
                    unique=sorted(set(g)-owned)
                    if unique:groups.append(unique);owned.update(unique)
                frames.append(groups)
            na=len(frames);nk=max(map(len,frames));nm=max(len(g) for frame in frames for g in frame)
            index=torch.zeros((na,nk,nm),device=q.device,dtype=torch.long)
            valid=torch.zeros((na,nk,nm),device=q.device,dtype=torch.bool)
            # Prepare CPU index arrays once, then perform one batched GPU gather/view.
            ci=np.zeros((na,nk,nm),np.int64);cm=np.zeros((na,nk,nm),bool)
            for i,frame in enumerate(frames):
                for j,g in enumerate(frame):ci[i,j,:len(g)]=g;cm[i,j,:len(g)]=True
            index=torch.from_numpy(ci).to(q.device);valid=torch.from_numpy(cm).to(q.device)
            row_index=torch.arange(na,device=q.device)[:,None,None]
            data[pfx+'source_frame_mask']=cm.any(-1)
            data[pfx+'source_frame_token_ids']=ci
            for name,value in source_states.items():
                atom_h=torch.stack([value[ix].mean(0) for ix in state['atom_positions']])
                norm_h=lm.model.norm(atom_h.to(all_hidden.dtype))
                logits_s=lm.output(norm_h).float()
                denominator=logits_s.logsumexp(-1)
                selected=logits_s[row_index,index].masked_fill(~valid,float('-inf'))
                frame_logp=selected.logsumexp(-1)-denominator[:,None]
                data[pfx+'source_'+name+'_frame_logp']=frame_logp.nan_to_num(neginf=-1e30).cpu().numpy()
                data[pfx+'source_'+name+'_target_logp']=frame_logp[:,0].cpu().numpy()
                data[pfx+'source_'+name+'_top_logp']=(logits_s.max(-1).values-denominator).cpu().numpy()
                data[pfx+'source_'+name+'_other_logp']=frame_logp[:,1:].max(-1).values.cpu().numpy()
                data[pfx+'source_'+name+'_coverage_logp']=frame_logp.logsumexp(-1).cpu().numpy()
            def angle(a,b):
                return torch.nn.functional.cosine_similarity(a,b,dim=-1).cpu().numpy()
            data[pfx+'source_cosine']=angle(ov,ot)
            data[pfx+'source_visual_norm']=ov.norm(dim=-1).cpu().numpy()
            data[pfx+'source_text_norm']=ot.norm(dim=-1).cpu().numpy()
            data[pfx+'source_visual_mass']=weights.sum(-1).mean(0).cpu().numpy()
            actual=buf.pop('actual_attention').float()
            error=(actual-ov-ot).norm()/actual.norm().clamp_min(1e-12)
            assert float(error)<.03,('Attention decomposition mismatch',float(error))
            data[pfx+'source_reconstruction_relative_error']=error.cpu().numpy()
'''
assert needle in source
source=source.replace(needle,inject+needle)
source=source.replace("        attn.register_forward_pre_hook(pre,with_kwargs=True)","        def capture_attention(module,inputs,out):\n            buf['actual_attention']=out[0][0,state['qpos']].detach()\n        attn.register_forward_hook(capture_attention)\n        attn.register_forward_pre_hook(pre,with_kwargs=True)")
source=source.replace("'version':1,'layers':layers", "'version':3,'semantic_dictionary':semantic_words,'readout':'passive image-token/nonimage-token value contributions; local residual views are NOT full modality-only counterfactuals','layers':layers")
source=source.replace("results/features/{args.split}","results/source_v3/{args.split}")
source=source.replace("results/feature_config.json","results/source_v3_config.json").replace("results/feature_config_{args.split}.json","results/source_v3_config_{args.split}.json")
(TASK/'code/extract_sources.py').write_text(source)
