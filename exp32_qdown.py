"""Soft fg-sink with CORRECTED subject box (robot is bottom-center)."""
import time
from pathlib import Path
import cloudpickle, numpy as np, torch
from fastvideo import VideoGenerator
MODEL="/home/hal-kaiqin/models/matrix-game-2.0-zelda-longlive"
_FV="/home/hal-kaiqin/FastVideo"
_ACT=f"{_FV}/examples/training/finetune/WanGame2.1_1.3b_i2v/actions_801/W.npy"
TGT=f"{_FV}/assets/third-person/combine/robot_zelda_scene.jpg"
OUT=Path("/home/hal-kaiqin/FastVideo_attninj/attn_injection_out/exp32_qdown")
H,W,NF=480,832,201
def w_setup(worker):
    from fastvideo.models.dits.matrixgame2.causal_model import CausalMatrixGame2SelfAttention
    from fastvideo.models.dits.matrixgame2.attn_injection import INJECTOR
    tf=worker.pipeline.get_module("transformer")
    for m in tf.modules():
        if isinstance(m,CausalMatrixGame2SelfAttention): m.sink_size=1;m.local_attn_size=6
    INJECTOR.prepare(tf); INJECTOR.grid_rows=30; INJECTOR.grid_cols=52
    INJECTOR.fg_box=(14,30,21,31); INJECTOR.fg_soft=True; INJECTOR.fg_sigma=(5.0,5.0)  # ROBOT bottom-center
    INJECTOR.start_fg_sink()
    return True
def w_cfg(worker,b,qs):
    from fastvideo.models.dits.matrixgame2.attn_injection import INJECTOR
    INJECTOR.fg_boost=b; INJECTOR.fg_qsupp=qs; return (b,qs)
def rpc(gen,fn,*a): return gen.executor.collective_rpc(cloudpickle.dumps(fn),args=a)
def main():
    gen=VideoGenerator.from_pretrained(MODEL,num_gpus=1,use_fsdp_inference=False,dit_cpu_offload=False,vae_cpu_offload=False,text_encoder_cpu_offload=True,pin_cpu_memory=False)
    rpc(gen,w_setup)
    raw=np.load(_ACT,allow_pickle=True).item()
    kb=torch.from_numpy(np.asarray(raw['keyboard'],np.float32)[:NF,:4]); mo=torch.from_numpy(np.asarray(raw['mouse'],np.float32)[:NF,:])
    grid=torch.tensor([(NF+3)//4,H//8,W//8])
    for b,qs in [(4,2),(4,3),(4,4),(5,3)]:
        rpc(gen,w_cfg,float(b),float(qs)); t0=time.time()
        gen.generate_video(prompt="",image_path=TGT,mouse_cond=mo.unsqueeze(0),keyboard_cond=kb.unsqueeze(0),grid_sizes=grid,num_frames=NF,height=H,width=W,num_inference_steps=4,seed=42,output_path=str(OUT/f"b{b}_q{qs}"),save_video=True)
        print(f"  [rollout] b{b}_q{qs}: {time.time()-t0:.1f}s",flush=True)
    print("[done] exp31",flush=True)
if __name__=="__main__": main()
