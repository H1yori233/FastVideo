import type { Bag } from "./types";

const GROUPS: { name: string; keys: string[] }[] = [
  {
    name: "Model & Init",
    keys: [
      "model_path",
      "pretrained_model_name_or_path",
      "real_score_model_path",
      "fake_score_model_path",
      "init_weights_from_safetensors",
    ],
  },
  {
    name: "Data",
    keys: [
      "data_path",
      "num_height",
      "num_width",
      "num_frames",
      "num_latent_t",
      "dataloader_num_workers",
      "training_cfg_rate",
      "flow_shift",
      "group_frame",
      "group_resolution",
    ],
  },
  {
    name: "Optimization",
    keys: [
      "learning_rate",
      "lr_scheduler",
      "lr_warmup_steps",
      "lr_num_cycles",
      "lr_power",
      "min_lr_ratio",
      "weight_decay",
      "betas",
      "max_grad_norm",
      "scale_lr",
    ],
  },
  {
    name: "Training loop",
    keys: [
      "max_train_steps",
      "num_train_epochs",
      "train_batch_size",
      "train_sp_batch_size",
      "gradient_accumulation_steps",
    ],
  },
  {
    name: "Memory & precision",
    keys: [
      "mixed_precision",
      "dit_precision",
      "vae_precision",
      "master_weight_type",
      "enable_gradient_checkpointing_type",
      "selective_checkpointing",
    ],
  },
  {
    name: "Parallelism",
    keys: [
      "num_gpus",
      "sp_size",
      "tp_size",
      "hsdp_replicate_dim",
      "hsdp_shard_dim",
      "fsdp_sharding_startegy",
    ],
  },
  {
    name: "EMA",
    keys: ["use_ema", "ema_decay", "ema_start_step"],
  },
  {
    name: "DMD distillation",
    keys: [
      "dmd_denoising_steps",
      "generator_update_interval",
      "min_timestep_ratio",
      "max_timestep_ratio",
      "real_score_guidance_scale",
      "fake_score_learning_rate",
      "fake_score_betas",
      "fake_score_lr_scheduler",
      "distill_cfg",
      "dfake_gen_update_ratio",
      "precondition_outputs",
    ],
  },
  {
    name: "Quantization",
    keys: ["generator_4bit_attn", "generator_4bit_linear", "override_text_encoder_quant"],
  },
  {
    name: "Validation",
    keys: [
      "validation_dataset_file",
      "validation_preprocessed_path",
      "validation_steps",
      "validation_sampling_steps",
      "validation_guidance_scale",
      "log_validation",
      "visualization_steps",
      "log_visualization",
    ],
  },
  {
    name: "Checkpointing",
    keys: [
      "output_dir",
      "training_state_checkpointing_steps",
      "weight_only_checkpointing_steps",
      "checkpoints_total_limit",
      "resume_from_checkpoint",
    ],
  },
  {
    name: "Tracking & misc",
    keys: ["tracker_project_name", "wandb_run_name", "trackers", "seed", "inference_mode"],
  },
];

function norm(key: string): string {
  return key.replace(/^--/, "").replace(/-/g, "_");
}

const KEY_TO_GROUP = new Map<string, string>();
for (const g of GROUPS) for (const k of g.keys) KEY_TO_GROUP.set(k, g.name);

export interface GroupedArgs {
  group: string;
  entries: [string, string][];
}

export function groupArgs(args: Bag): GroupedArgs[] {
  const byGroup = new Map<string, [string, string][]>();
  for (const g of GROUPS) byGroup.set(g.name, []);
  for (const [rawKey, v] of Object.entries(args)) {
    const k = norm(rawKey);
    const group = KEY_TO_GROUP.get(k) ?? "Tracking & misc";
    byGroup.get(group)!.push([k, String(v)]);
  }
  return GROUPS.map((g) => {
    const entries = byGroup.get(g.name)!;
    const pos = new Map(g.keys.map((k, i) => [k, i]));
    entries.sort((a, b) => (pos.get(a[0]) ?? 99) - (pos.get(b[0]) ?? 99));
    return { group: g.name, entries };
  }).filter((g) => g.entries.length > 0);
}

export function normalizeArgs(args: Bag): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(args)) out[norm(k)] = String(v);
  return out;
}

export function groupOf(key: string): string {
  return KEY_TO_GROUP.get(norm(key)) ?? "Tracking & misc";
}
