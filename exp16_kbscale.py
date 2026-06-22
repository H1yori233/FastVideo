import time
from pathlib import Path
import cloudpickle, numpy as np, torch
from fastvideo import VideoGenerator
MODEL="/home/hal-kaiqin/models/matrix-game-2.0-zelda-longlive"
_FV="/home/hal-kaiqin/FastVideo"
_ACT=f"{_FV}/examples/training/finetune/WanGame2.1_1.3b_i2v/actions_801/W.npy"
TGT=f"{_FV}/assets/third-person/combine/robot_zelda_scene.jpg"
OUT=Path("/home/hal-kaiqin/FastVideo_attninj/attn_injection_out/exp16_kbscale")
H,W,NF=480,832,117
SINK=3
SCALES=[1.0,2.0,3.0,4.0]
def w_set(worker,sink,window):
    from fastvideo.models.dits.matrixgame2.causal_model import CausalMatrixGame2SelfAttention
    tf=worker.pipeline.get_module("transformer"); n=0
    for m in tf.modules():
        if isinstance(m,CausalMatrixGame2SelfAttention): m.sink_size=sink;m.local_attn_size=window;n+=1
    tf.local_attn_size=window
    for st in worker.pipeline.stages:
        if hasattr(st,"local_attn_size"): st.local_attn_size=window
    return (sink,window,n)
def rpc(gen,fn,*a): return gen.executor.collective_rpc(cloudpickle.dumps(fn),args=a)
def main():
    print(f"[init] {MODEL}",flush=True)
    gen=VideoGenerator.from_pretrained(MODEL,num_gpus=1,use_fsdp_inference=False,dit_cpu_offload=False,vae_cpu_offload=False,text_encoder_cpu_offload=True,pin_cpu_memory=False)
    print(f"[set] {rpc(gen,w_set,SINK,6)[0]}",flush=True)
    raw=np.load(_ACT,allow_pickle=True).item()
    kb0=np.asarray(raw["keyboard"],np.float32)[:NF,:4]; mo=torch.from_numpy(np.asarray(raw["mouse"],np.float32)[:NF,:])
    grid=torch.tensor([(NF+3)//4,H//8,W//8])
    for sc in SCALES:
        kb=torch.from_numpy(kb0*sc); t0=time.time()
        gen.generate_video(prompt="",image_path=TGT,mouse_cond=mo.unsqueeze(0),keyboard_cond=kb.unsqueeze(0),grid_sizes=grid,num_frames=NF,height=H,width=W,num_inference_steps=4,seed=42,output_path=str(OUT/f"kb{int(sc)}x"),save_video=True)
        print(f"  [rollout] kb{int(sc)}x: {time.time()-t0:.1f}s",flush=True)
    print("[done] exp16",flush=True)
if __name__=="__main__":
    main()
