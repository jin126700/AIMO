"""FP32 수치 검증, 별도 행동 cohort 및 prompt Page."""
import dataclasses,json,time,sys,os,gc
from pathlib import Path
import torch
from .fp32_runtime import precision,config,policy,Guard,memory_settings
from .real_run import sha
from .runtime import atomic_write_json as write,DedupLedger,store_lock
from .bf16_collect import identity,slot_path
from .bf16_batch import BatchedQwen
from .adapters.qwen import load_real_qwen,SlotRequest,classify_thinking_slot,SCORER_VERSION

def load(root):
    torch.set_num_threads(2);settings=precision();p=policy(root);cfg=config(root)
    free,total=torch.cuda.mem_get_info()
    groups=json.loads((root/'collection/groups.json').read_text())
    prompt_max=max(len(pr['input_ids']) for g in groups for pr in g['prompts'])
    mem=memory_settings(root)
    batch=mem['batch_size']
    required=16_500_000_000+batch*(16384+prompt_max)*36*2*8*128*4+8*1024**3
    if batch==1:
        assert os.environ.get('PYTORCH_CUDA_ALLOC_CONF')=='expandable_segments:True'
    assert free>required,(free,required)
    torch.cuda.set_per_process_memory_fraction(mem['allocator_fraction'])
    model,tok=load_real_qwen(cfg)
    assert {v.dtype for v in model.parameters() if v.is_floating_point()}=={torch.float32}
    assert not getattr(model,'is_quantized',False)
    assert sha(tok.chat_template)==p['observation_provenance']['template_hash']
    assert sha(tok.backend_tokenizer.to_str())==p['observation_provenance']['tokenizer_hash']
    settings.update(parameter_dtypes=['torch.float32'],gpu=torch.cuda.get_device_name(),
                    free_vram_before=free,estimated_required_bytes=required,
                    attention_backend=model.config._attn_implementation,execution_memory=mem,
                    allocator_environment=os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),model_parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),model_sharding=False)
    return model,tok,BatchedQwen(cfg.server.thinking,model,tok),settings

def request(root,model,tok,backend,g,pr):
    from .adapters.qwen import ExtractionRequest,select_landmarks
    rendered=backend.render(pr['prompt']);assert rendered['input_ids'][0].tolist()==pr['input_ids']
    text=rendered['text'];start=text.index(pr['prompt']);end=start+len(pr['prompt'])
    spans=tok(text,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
    body=[i for i,(a,b) in enumerate(spans) if a>=start and b<=end and b>a]
    offsets,valid,rel=select_landmarks([body[round(j*(len(body)-1)/15)] for j in range(16)],len(spans))
    return ExtractionRequest(g['original_id'],pr['prompt_id'],rendered['input_ids'].to('cuda'),offsets,valid,rel)

def extract(root,model,tok,backend,g,pr):
    from .adapters.qwen import extract_page
    req=request(root,model,tok,backend,g,pr)
    prov={**policy(root)['observation_provenance'],'input_token_hash':pr['input_token_hash'],
          'precision_manifest_sha256':sha((root/'protocol/precision_manifest.json').read_bytes())}
    return extract_page(model,req.input_ids,req.landmark_offsets,req.valid,req.relative_positions,
                        req.original_id,req.variant_id,provenance=prov)

def audit(root):
    model,tok,backend,settings=load(root)
    g=json.loads((root/'calibration/groups.json').read_text())[0];pr=g['prompts'][0]
    ids=backend.render(pr['prompt'])['input_ids'].to('cuda')
    activations={}
    def hook(name):
        def capture(module,args,output):
            tensor=output[0] if isinstance(output,tuple) else output
            assert tensor.dtype==torch.float32
            activations[name]=str(tensor.dtype)
        return capture
    handles=[model.model.layers[i].register_forward_hook(hook(str(i))) for i in [0,18,35]]
    with torch.inference_mode():
        output=model(input_ids=ids,use_cache=True,logits_to_keep=1)
    cache=output.past_key_values
    if hasattr(cache,'layers'):
        dtypes={str(x.dtype) for layer in cache.layers for x in [layer.keys,layer.values]}
    else:dtypes={str(x.dtype) for layer in cache for x in layer}
    assert dtypes=={'torch.float32'}
    settings.update(activation_dtypes=activations,kv_cache_dtypes=sorted(dtypes),logits_dtype=str(output.logits.dtype))
    for h in handles:h.remove()
    del output,cache
    write(root/'protocol/precision_manifest.json',settings)
    with torch.inference_mode():
        a=extract(root,model,tok,backend,g,pr);b=extract(root,model,tok,backend,g,pr)
    noise=max(float((a.state-b.state).abs().max()),float((a.updates-b.updates).abs().max()))
    assert noise<=1e-5 and a.residual_identity_error()<1e-3
    assert a.state.dtype==a.updates.dtype==torch.float32
    # 별도 기술 요청만 token cap과 stop 시험에 사용합니다.
    reqs=[SlotRequest('technical',f'technical-{i}','Compute 2 + 2.','4',12345+i) for i in range(2)]
    x=backend.generate_batch(reqs,max_tokens=8);y=backend.generate_batch(reqs,max_tokens=8)
    assert [r.generated_token_ids for r in x]==[r.generated_token_ids for r in y]
    calls=[0]
    def stop():calls[0]+=1;return calls[0]>=3
    backend.stop_check=stop
    stopped=backend.generate_batch(reqs,max_tokens=8)
    assert all(r.termination_reason=='external_stop' for r in stopped)
    from .scoring import compare_answers
    assert compare_answers('4','4')=='correct' and compare_answers('5','4')=='wrong'
    # 실제 thinking parser의 정답/오답/자연 cap 분리.
    checks={}
    for ans in ['4','5']:
        checks[ans]=classify_thinking_slot(started=True,infra_error=False,hit_cap=False,is_final_cap=False,
            text='reasoning</think>\\n\\boxed{'+ans+'}',gold='4',thinking_already_open=True)
    assert checks=={'4':'C','5':'W'},checks
    write(root/'calibration/numerical_audit.json',{'repeat_max_abs':noise,'residual_error':a.residual_identity_error(),
          'state_dtype':str(a.state.dtype),'update_dtype':str(a.updates.dtype),'seed_repeat_identical':True,
          'technical_stop_passed':True,'scorer_checks':checks,'calibration_untouched':True,
          'peak_vram_bytes':torch.cuda.max_memory_allocated()})
    print(json.dumps(settings),flush=True)

def collect(root,stage,g,pr,backend):
    pol=policy(root);guard=Guard(root);ledger=DedupLedger.open(root/stage,'slots.jsonl')
    mem=memory_settings(root)
    assert backend.render(pr['prompt'])['input_ids'][0].tolist()==pr['input_ids']
    with store_lock(root/stage/(sha(g['original_id'])+'.group')):
        pending=[]
        for s in pr['slots']:
            path=slot_path(root,stage,s['slot_id']);ident=identity(pr,s,pol)
            if path.exists():
                raw=json.loads(path.read_text());assert raw['identity']==ident
                if raw['state']=='started':
                    raw.update(state='interrupted',outcome='U_score',termination='interrupted')
                    write(path,raw)
                if not ledger.seen(s['slot_id']):
                    ledger.mark(s['slot_id'],{'artifact':str(path),'outcome':raw['outcome'],'identity':ident})
                continue
            assert not ledger.seen(s['slot_id'])
            pending.append((s,path,ident))
        for offset in range(0,len(pending),mem['batch_size']):
            if guard.should_stop()[0]:return
            batch=pending[offset:offset+mem['batch_size']]
            start=time.monotonic();torch.cuda.reset_peak_memory_stats()
            before_allocated=torch.cuda.memory_allocated()
            cache_info={'max_cache_bytes':0,'kv_dtypes':[],'logits_dtype':None}
            results=[]
            for s,path,ident in batch:
                write(path,{'state':'started','identity':ident,'original_id':g['original_id'],
                      'prompt_id':pr['prompt_id'],'started_utc':time.time(),
                      'execution_memory_id':mem['execution_id'],'batch_size':len(batch)})
            lookup={s['slot_id']:(path,ident) for s,path,ident in batch}
            def save(result):
                path,ident=lookup[result.slot_id]
                outcome=classify_thinking_slot(started=result.started,infra_error=result.infra_error,hit_cap=result.hit_cap,
                    is_final_cap=False,text=result.text,gold=pr['gold'],thinking_already_open=result.thinking_already_open,
                    generated_token_ids=result.generated_token_ids,special_token_ids=backend.special_token_ids)
                interrupted=result.termination_reason in ('external_stop','infra_error')
                if interrupted:outcome='U_score'
                raw={'identity':ident,'state':'interrupted' if interrupted else 'completed',
                     'original_id':g['original_id'],'prompt_id':pr['prompt_id'],'outcome':outcome,
                     'termination':result.termination_reason,'result':dataclasses.asdict(result),
                     'scorer':{'gold':pr['gold'],'version':SCORER_VERSION,'outcome':outcome},
                     'batch_elapsed_seconds':time.monotonic()-start,'completed_utc':time.time(),
                     'execution_memory_id':mem['execution_id'],'batch_size':len(batch)}
                assert result.input_token_ids==pr['input_ids']
                write(path,raw);ledger.mark(result.slot_id,{'artifact':str(path),'outcome':outcome,'identity':ident})
                results.append(result)
            def observe(cache,logits,step,n):
                tensors=([x for layer in cache.layers for x in [layer.keys,layer.values]]
                         if hasattr(cache,'layers') else [x for layer in cache for x in layer])
                dtypes={str(x.dtype) for x in tensors}
                assert dtypes=={'torch.float32'} and logits.dtype==torch.float32
                assert not torch.is_autocast_enabled() and not torch.backends.cuda.matmul.allow_tf32
                size=sum(x.numel()*x.element_size() for x in tensors)
                cache_info.update(max_cache_bytes=max(cache_info['max_cache_bytes'],size),
                                  kv_dtypes=sorted(dtypes),logits_dtype=str(logits.dtype),cache_class=type(cache).__name__)
                write(root/'runtime'/('progress_0_'+str(os.getpid())+'.json'),
                      {'stage':stage,'prompt_id':pr['prompt_id'],'step':step,'batch':n,'time':time.time(),
                       'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),
                       'memory_allocated':torch.cuda.memory_allocated(),'memory_reserved':torch.cuda.memory_reserved(),
                       'execution_memory_id':mem['execution_id'],**cache_info})
            backend.stop_check=lambda:guard.should_stop()[0]
            backend.progress_callback=None
            backend.cache_observer=observe
            try:
                backend.generate_batch([SlotRequest(pr['prompt_id'],s['slot_id'],pr['prompt'],'',s['seed']) for s,_,_ in batch],on_result=save)
            except torch.OutOfMemoryError:
                (root/'runtime'/('oom_memory_'+str(os.getpid())+'.txt')).write_text(torch.cuda.memory_summary(abbreviated=True))
                raise
            finally:
                peak=torch.cuda.max_memory_allocated();peak_reserved=torch.cuda.max_memory_reserved()
                backend.cache_observer=None
                gc.collect();torch.cuda.empty_cache()
            elapsed=time.monotonic()-start
            report={'original_id':g['original_id'],'prompt_id':pr['prompt_id'],'slot_ids':[s['slot_id'] for s,_,_ in batch],
                    'batch_size':len(batch),'wall_seconds':elapsed,'peak_vram_bytes':peak,
                    'peak_reserved_bytes':peak_reserved,'before_allocated_bytes':before_allocated,
                    'after_allocated_bytes':torch.cuda.memory_allocated(),'after_reserved_bytes':torch.cuda.memory_reserved(),
                    'inactive_split_bytes':torch.cuda.memory_stats().get('inactive_split_bytes.all.current',0),
                    'generated_tokens':sum(r.generated_tokens for r in results),
                    'tokens_per_second':sum(r.generated_tokens for r in results)/elapsed,
                    'execution_memory_id':mem['execution_id'],'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),
                    'policy_hash':pol['behavior_policy_id'],**cache_info}
            write(root/'runtime/memory_batches'/(sha(batch[0][0]['slot_id'])+'.json'),report)
            assert report['after_allocated_bytes']<=before_allocated+64*1024**2,'cache not released'

def worker(root,stage,index,group_indices=None):
    model,tok,backend,settings=load(root)
    write(root/stage/f"precision_gpu{index}_{memory_settings(root)['execution_id']}.json",settings)
    assert sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in model.modules())==0
    groups=json.loads((root/stage/'groups.json').read_text())
    selected=[groups[i] for i in group_indices] if group_indices is not None else groups[index::4]
    for g in selected:
        for pr in g['prompts']:
            if Guard(root).should_stop()[0]:return
            collect(root,stage,g,pr,backend)
            torch.cuda.empty_cache()
        print(json.dumps({'stage':stage,'original_id':g['original_id'],'time':time.time()}),flush=True)

def pages(root):
    from .page import save_pages,load_pages
    start=time.monotonic()
    index=json.loads((root/'pages/index.json').read_text()) if (root/'pages/index.json').exists() else {}
    groups=json.loads((root/'collection/groups.json').read_text())
    expected=[p for g in groups for p in g['prompts']]
    if all(p['prompt_id'] in index for p in expected):
        for pr in expected:
            item=index[pr['prompt_id']]
            assert sha(Path(item['path']).read_bytes())==item['sha256']
            assert item['input_token_hash']==pr['input_token_hash']
        write(root/'pages/oom_resume_cache_check.json',{'verified':len(expected),'model_loaded':False})
        return
    model,tok,backend,settings=load(root)
    write(root/'pages/precision.json',settings)
    for g in groups:
        for pr in g['prompts']:
            if Guard(root).should_stop()[0]:raise RuntimeError('Page extraction stopped')
            if pr['prompt_id'] in index:
                item=index[pr['prompt_id']]
                assert sha(Path(item['path']).read_bytes())==item['sha256']
                continue
            with torch.inference_mode():page=extract(root,model,tok,backend,g,pr)
            path=root/'pages/artifacts'/(sha(pr['prompt_id'])+'.npz');path.parent.mkdir(exist_ok=True)
            save_pages([page],path)
            if len(index)<2:
                from .fp32_runtime import PARENT
                old_index=json.loads((PARENT/'pages/index.json').read_text())
                if pr['prompt_id'] in old_index:
                    old_item=old_index[pr['prompt_id']]
                    assert sha(Path(old_item['path']).read_bytes())==old_item['sha256']
                    old_page=load_pages(old_item['path'])[0]
                    write(root/'pages'/('old_comparison_'+str(len(index))+'.json'),
                          {'prompt_id':pr['prompt_id'],'old_path':old_item['path'],
                           'state_max_abs':float((page.state-old_page.state).abs().max()),
                           'update_max_abs':float((page.updates-old_page.updates).abs().max()),
                           'old_precision_unverified':True,'old_metadata_preserved':True,'reused':False})
            assert load_pages(path)[0].content_fingerprint()==page.content_fingerprint()
            index[pr['prompt_id']]={'path':str(path),'sha256':sha(path.read_bytes()),'input_token_hash':pr['input_token_hash'],
                'original_id':g['original_id'],'content_fingerprint':page.content_fingerprint(),'split':g['split'],'reused':False}
            write(root/'pages/index.json',index)
    write(root/'pages/summary.json',{'new_main_pages':len(index),'reused_pages':0,'model_config_hash':page.provenance['config_hash'] if 'page' in locals() else load_pages(Path(next(iter(index.values()))['path']))[0].provenance['config_hash'],
          'wall_seconds':time.monotonic()-start,'peak_vram_bytes':torch.cuda.max_memory_allocated(),'dtype':'float32'})

if __name__=='__main__':
    mode=sys.argv[1];root=Path(sys.argv[2])
    if mode=='audit':audit(root)
    elif mode=='worker':worker(root,sys.argv[3],int(sys.argv[4]))
    elif mode=='pages':pages(root)
    elif mode=='smoke':
        plan=json.loads((root/'runtime/oom_recovery_001/smoke_plan.json').read_text())
        worker(root,'collection',plan['gpu'],plan['group_indices'])
