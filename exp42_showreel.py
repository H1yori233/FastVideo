import time
from pathlib import Path
import cloudpickle, numpy as np, torch
from fastvideo import VideoGenerator
MODEL="/home/hal-kaiqin/models/matrix-game-2.0-zelda-longlive"
_FV="/home/hal-kaiqin/FastVideo"; _C=f"{_FV}/assets/third-person/combine"; _T=f"{_FV}/assets/third-person"
_ACT=f"{_FV}/examples/training/finetune/WanGame2.1_1.3b_i2v/actions_801/W.npy"
OUT=Path("/home/hal-kaiqin/FastVideo_attninj/attn_injection_out/exp42_show")
H,W,NF=480,832,117
def w_fg(worker,box):
    from fastvideo.models.dits.matrixgame2.causal_model import CausalMatrixGame2SelfAttention
    from fastvideo.models.dits.matrixgame2.attn_injection import INJECTOR
    tf=worker.pipeline.get_module("transformer")
    for m in tf.modules():
        if isinstance(m,CausalMatrixGame2SelfAttention): m.sink_size=1;m.local_attn_size=6
    INJECTOR.prepare(tf); INJECTOR.grid_rows=30; INJECTOR.grid_cols=52
    INJECTOR.fg_box=tuple(box); INJECTOR.fg_soft=True; INJECTOR.fg_sigma=(5.0,5.0)
    INJECTOR.start_fg_sink(); INJECTOR.fg_boost=4.0; INJECTOR.fg_qsupp=6.0
    INJECTOR.layer_lo=0.2; INJECTOR.layer_hi=0.8
    return box
def w_off(worker):
    from fastvideo.models.dits.matrixgame2.causal_model import CausalMatrixGame2SelfAttention
    from fastvideo.models.dits.matrixgame2.attn_injection import INJECTOR
    tf=worker.pipeline.get_module("transformer")
    for m in tf.modules():
        if isinstance(m,CausalMatrixGame2SelfAttention): m.sink_size=0;m.local_attn_size=6
    INJECTOR.off(); return True
def rpc(gen,fn,*a): return gen.executor.collective_rpc(cloudpickle.dumps(fn),args=a)
def main():
    gen=VideoGenerator.from_pretrained(MODEL,num_gpus=1,use_fsdp_inference=False,dit_cpu_offload=False,vae_cpu_offload=False,text_encoder_cpu_offload=True,pin_cpu_memory=False)
    raw=np.load(_ACT,allow_pickle=True).item()
    kb=torch.from_numpy(np.asarray(raw['keyboard'],np.float32)[:NF,:4]); mo=torch.from_numpy(np.asarray(raw['mouse'],np.float32)[:NF,:])
    grid=torch.tensor([(NF+3)//4,H//8,W//8])
    def run(img,tag):
        t0=time.time()
        gen.generate_video(prompt="",image_path=img,mouse_cond=mo.unsqueeze(0),keyboard_cond=kb.unsqueeze(0),grid_sizes=grid,num_frames=NF,height=H,width=W,num_inference_steps=4,seed=42,output_path=str(OUT/tag),save_video=True)
        print(f"  [rollout] {tag}: {time.time()-t0:.1f}s",flush=True)
    rpc(gen,w_fg,(12,27,23,32)); run(f"{_T}/genshin.png","genshin_fg")
    rpc(gen,w_fg,(12,27,23,33)); run(f"{_T}/mc_third_person.jpg","mc_fg")
    rpc(gen,w_off); run(f"{_C}/robot_dog_zelda_scene.jpg","dog_clean")
    print("[done] exp42",flush=True)
if __name__=="__main__": main()
