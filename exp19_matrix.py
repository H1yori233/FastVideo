"""Snow-scene x action matrix with the validated recipe: sink3 + action*3."""
import time
from pathlib import Path
import cloudpickle, numpy as np, torch
from fastvideo import VideoGenerator
MODEL="/home/hal-kaiqin/models/matrix-game-2.0-zelda-longlive"
_FV="/home/hal-kaiqin/FastVideo"
_ACTD=f"{_FV}/examples/training/finetune/WanGame2.1_1.3b_i2v/actions_801"
_COMB=f"{_FV}/assets/third-person/combine"
OUT=Path("/home/hal-kaiqin/FastVideo_attninj/attn_injection_out/exp19_matrix")
H,W,NF=480,832,117
SCALE=3.0
SCENES={"robot":f"{_COMB}/robot_zelda_scene.jpg","robotdog":f"{_COMB}/robot_dog_zelda_scene.jpg",
        "wukong":f"{_COMB}/wukong_zelda_scene.jpg","car":f"{_COMB}/car_zelda_scene.jpg"}
ACTIONS=["W","r","A"]   # forward-run, yaw-right, strafe-left
def w_set(worker):
    from fastvideo.models.dits.matrixgame2.causal_model import CausalMatrixGame2SelfAttention
    tf=worker.pipeline.get_module("transformer")
    for m in tf.modules():
        if isinstance(m,CausalMatrixGame2SelfAttention): m.sink_size=3;m.local_attn_size=6
    return True
def rpc(gen,fn,*a): return gen.executor.collective_rpc(cloudpickle.dumps(fn),args=a)
def load_act(name):
    raw=np.load(f"{_ACTD}/{name}.npy",allow_pickle=True).item()
    kb=np.asarray(raw['keyboard'],np.float32)[:NF,:4]*SCALE
    mo=np.asarray(raw['mouse'],np.float32)[:NF,:]*SCALE
    return torch.from_numpy(kb),torch.from_numpy(mo)
def main():
    gen=VideoGenerator.from_pretrained(MODEL,num_gpus=1,use_fsdp_inference=False,dit_cpu_offload=False,vae_cpu_offload=False,text_encoder_cpu_offload=True,pin_cpu_memory=False)
    rpc(gen,w_set); print("[set] sink3",flush=True)
    grid=torch.tensor([(NF+3)//4,H//8,W//8])
    for sc,img in SCENES.items():
        for act in ACTIONS:
            kb,mo=load_act(act); t0=time.time()
            gen.generate_video(prompt="",image_path=img,mouse_cond=mo.unsqueeze(0),keyboard_cond=kb.unsqueeze(0),grid_sizes=grid,num_frames=NF,height=H,width=W,num_inference_steps=4,seed=42,output_path=str(OUT/f"{sc}_{act}"),save_video=True)
            print(f"  [rollout] {sc}_{act}: {time.time()-t0:.1f}s",flush=True)
    print("[done] exp19",flush=True)
if __name__=="__main__": main()
