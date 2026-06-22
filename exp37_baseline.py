import time
from pathlib import Path
import cloudpickle, numpy as np, torch
from fastvideo import VideoGenerator
MODEL="/home/hal-kaiqin/models/matrix-game-2.0-zelda-longlive"
_FV="/home/hal-kaiqin/FastVideo"
_ACT=f"{_FV}/examples/training/finetune/WanGame2.1_1.3b_i2v/actions_801/W.npy"
TGT=f"{_FV}/assets/third-person/combine/robot_zelda_scene.jpg"
OUT=Path("/home/hal-kaiqin/FastVideo_attninj/attn_injection_out/exp37_baseline")
H,W,NF=480,832,201
def w_off(worker):
    from fastvideo.models.dits.matrixgame2.causal_model import CausalMatrixGame2SelfAttention
    from fastvideo.models.dits.matrixgame2.attn_injection import INJECTOR
    tf=worker.pipeline.get_module("transformer")
    for m in tf.modules():
        if isinstance(m,CausalMatrixGame2SelfAttention): m.sink_size=0;m.local_attn_size=6
    INJECTOR.off()
    return True
def rpc(gen,fn,*a): return gen.executor.collective_rpc(cloudpickle.dumps(fn),args=a)
def main():
    gen=VideoGenerator.from_pretrained(MODEL,num_gpus=1,use_fsdp_inference=False,dit_cpu_offload=False,vae_cpu_offload=False,text_encoder_cpu_offload=True,pin_cpu_memory=False)
    rpc(gen,w_off)
    raw=np.load(_ACT,allow_pickle=True).item()
    kb=torch.from_numpy(np.asarray(raw['keyboard'],np.float32)[:NF,:4]); mo=torch.from_numpy(np.asarray(raw['mouse'],np.float32)[:NF,:])
    grid=torch.tensor([(NF+3)//4,H//8,W//8])
    gen.generate_video(prompt="",image_path=TGT,mouse_cond=mo.unsqueeze(0),keyboard_cond=kb.unsqueeze(0),grid_sizes=grid,num_frames=NF,height=H,width=W,num_inference_steps=4,seed=42,output_path=str(OUT/"clean"),save_video=True)
    print("[done] exp37",flush=True)
if __name__=="__main__": main()
