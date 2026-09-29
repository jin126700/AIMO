"""Native Qwen batched decoding with independent per-request CUDA generators."""
import time
import torch
from .adapters.qwen import QwenGenerationBackend, SlotResult,check_total_context

class BatchedQwen(QwenGenerationBackend):
    version='native_dynamiccache_v1'
    @torch.inference_mode()
    def generate_batch(self,requests,on_result=lambda r:None,max_tokens=None):
        from transformers.generation.logits_process import TemperatureLogitsWarper,TopKLogitsWarper,TopPLogitsWarper
        limit=max_tokens or self.profile.max_new_tokens
        device=next(self.model.parameters()).device
        renders=[self.render(r.prompt) for r in requests]
        tokens=[r['input_ids'][0].tolist() for r in renders]
        for t in tokens:check_total_context(len(t),limit,self.profile.max_total_context)
        width=max(map(len,tokens));n=len(tokens)
        pad=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        ids=torch.tensor([[pad]*(width-len(t))+t for t in tokens],device=device)
        mask=torch.tensor([[0]*(width-len(t))+[1]*len(t) for t in tokens],device=device)
        generators=[torch.Generator(device=device).manual_seed(r.seed) for r in requests]
        generated=torch.full((n,limit),pad,dtype=torch.long,device=device)
        eos=self.model.generation_config.eos_token_id
        eos=set(eos if isinstance(eos,list) else [eos]);done=set();past=None;results=[]
        warpers=[TemperatureLogitsWarper(self.profile.temperature),TopKLogitsWarper(self.profile.top_k),TopPLogitsWarper(self.profile.top_p)]
        assert self.profile.min_p==0 and self.profile.do_sample
        def finish(i,steps,reason,error=None):
            out=generated[i,:steps].cpu().tolist()
            result=SlotResult(requests[i].slot_id,text=self.tokenizer.decode(out,skip_special_tokens=False),prompt_tokens=len(tokens[i]),generated_tokens=len(out),generated_token_ids=out,input_token_ids=tokens[i],hit_cap=reason=='token_cap',termination_reason=reason,infra_error=reason=='infra_error',error_message=error,thinking_already_open=renders[i]['text'].rstrip().endswith('<think>'))
            on_result(result);results.append(result);done.add(i)
        step=0
        try:
            for step in range(limit):
                if self.stop_check():
                    for i in range(n):
                        if i not in done:finish(i,step,'external_stop')
                    break
                positions=mask.long().cumsum(-1)-1;positions.masked_fill_(mask==0,1)
                output=self.model(input_ids=ids,attention_mask=mask,position_ids=positions if past is None else positions[:,-1:],past_key_values=past,use_cache=True,logits_to_keep=1,output_hidden_states=False,output_attentions=False)
                past=output.past_key_values
                if step % 128 == 0 and getattr(self,'progress_callback',None):
                    self.progress_callback(step, n)
                if step % 128 == 0 and getattr(self,'cache_observer',None):
                    self.cache_observer(past, output.logits, step, n)
                scores=output.logits[:,-1,:].float()
                for warp in warpers:scores=warp(None,scores)
                probs=scores.softmax(-1)
                nxt=torch.stack([torch.multinomial(probs[i],1,generator=generators[i])[0] if i not in done else torch.tensor(pad,device=device) for i in range(n)])
                generated[:,step]=nxt
                # Only batch-size token IDs cross to CPU, never a vocabulary tensor.
                for i,tok in enumerate(nxt.tolist()):
                    if i not in done and (tok in eos or step+1==limit):finish(i,step+1,'eos' if tok in eos else 'token_cap')
                if len(done)==n:break
                ids=nxt[:,None];mask=torch.cat((mask,torch.ones((n,1),device=device,dtype=mask.dtype)),dim=1)
        except Exception as exc:
            for i in range(n):
                if i not in done:finish(i,step,'infra_error',repr(exc))
            raise
        finally:
            del past
        return results
