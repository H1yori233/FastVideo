"""In-rollout slow-drift correction: EMA of block latents, subtract drift toward
anchor each block (identity), keep fast residual (motion). sink0 + kb3x drive."""
import time, sys
from pathlib import Path
import cloudpickle, numpy as np, torch
from fastvideo import VideoGenerator
MODEL="/home/hal-kaiqin/models/matrix-game-2.0-zelda-longlive"
_FV="/home/hal-kaiqin/FastVideo"
_ACTD=f"{_FV}/examples/training/finetune/WanGame2.1_1.3b_i2v/actions_801"
TGT=f"{_FV}/assets/third-person/combine/robot_zelda_scene.jpg"
OUT=Path("/home/hal-kaiqin/FastVideo_attninj/attn_injection_out/exp18_drift")
H,W,NF=480,832,117
KBSCALE=3.0
GAMMAS=[0.0,0.3,0.5,0.7]

def w_setup(worker):
    from fastvideo.models.dits.matrixgame2.causal_model import CausalMatrixGame2SelfAttention
    tf=worker.pipeline.get_module("transformer")
    for m in tf.modules():
        if isinstance(m,CausalMatrixGame2SelfAttention): m.sink_size=0; m.local_attn_size=6
    tf.local_attn_size=6
    for st in worker.pipeline.stages:
        if hasattr(st,"local_attn_size"): st.local_attn_size=6
    return True

def w_install_drift(worker, beta, gamma):
    stage=None
    for st in worker.pipeline.stages:
        if hasattr(st,'local_attn_size') and hasattr(st,'_update_context_cache'):
            stage=st; break
    if stage is None:
        raise RuntimeError("denoising stage not found: "+str([type(x).__name__ for x in worker.pipeline.stages]))
    if getattr(stage,'_orig_ucc',None) is None:
        stage._orig_ucc=stage._update_context_cache
    orig=stage._orig_ucc
    state={"m":None,"m0":None}
    def patched(current_latents, batch, start_index, current_num_frames, ctx, action_kwargs, context_noise):
        z=current_latents
        if start_index==0:
            state["m0"]=z.detach().clone(); state["m"]=z.detach().clone()
        elif gamma>0:
            state["m"]=beta*state["m"]+(1.0-beta)*z.detach()
            z=z-gamma*(state["m"]-state["m0"])
            batch.latents[:,:,start_index:start_index+current_num_frames,:,:]=z
        return orig(current_latents=z, batch=batch, start_index=start_index, current_num_frames=current_num_frames, ctx=ctx, action_kwargs=action_kwargs, context_noise=context_noise)
    stage._update_context_cache=patched
    return (beta,gamma)

def rpc(gen,fn,*a): return gen.executor.collective_rpc(cloudpickle.dumps(fn),args=a)

def main():
    gen=VideoGenerator.from_pretrained(MODEL,num_gpus=1,use_fsdp_inference=False,dit_cpu_offload=False,vae_cpu_offload=False,text_encoder_cpu_offload=True,pin_cpu_memory=False)
    rpc(gen,w_setup); print("[setup] sink0 done",flush=True)
    raw=np.load(f"{_ACTD}/W.npy",allow_pickle=True).item()
    kb=torch.from_numpy(np.asarray(raw['keyboard'],np.float32)[:NF,:4]*KBSCALE)
    mo=torch.from_numpy(np.asarray(raw['mouse'],np.float32)[:NF,:])
    grid=torch.tensor([(NF+3)//4,H//8,W//8])
    def run(tag):
        t0=time.time()
        gen.generate_video(prompt="",image_path=TGT,mouse_cond=mo.unsqueeze(0),keyboard_cond=kb.unsqueeze(0),grid_sizes=grid,num_frames=NF,height=H,width=W,num_inference_steps=4,seed=42,output_path=str(OUT/tag),save_video=True)
        print(f"  [rollout] {tag}: {time.time()-t0:.1f}s",flush=True)
    # g=0 first (no patch) -> also builds pipeline stages lazily
    run("g000")
    for g in [x for x in GAMMAS if x>0]:
        print(f"[install] {rpc(gen,w_install_drift,0.9,g)[0]}",flush=True)
        run(f"g{int(g*100):03d}")
    print("[done] exp18",flush=True)
if __name__=="__main__": main()
