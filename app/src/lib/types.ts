export type Bag = Record<string, unknown>;

export interface Project {
  id: string;
  name: string;
}

export interface Pipeline {
  id: string;
  project_id: string;
  name: string;
  kind?: string;
}

export interface BaseModel {
  id: string;
  name: string;
  hf_repo: string;
}

export interface DatasetSnapshot {
  id: string;
  name: string;
  path: string;
}

export interface Experiment {
  id: string;
  name: string;
  project_id: string;
  pipeline_id: string;
  base_model_id: string;
  dataset_id: string;
  init_from: string;
  wandb_run_id: string | null;
  in_grid: boolean;
  training_args: Bag;
  environment: Bag;
  code: Bag;
}

export interface LineageEdge {
  from: string;
  to: string;
  kind: "init_weights" | "resume";
  step: number;
}

export interface Checkpoint {
  id: string;
  experiment_id: string;
  step: number;
  path: string;
  pending?: boolean;
}

export interface ValidationCase {
  id: string;
  index: number;
  caption: string;
  has_real_caption: boolean;
  resolution?: string | null;
}

export interface OutputVideo {
  id: string;
  experiment_id: string;
  checkpoint_id: string;
  validation_case_id: string;
  sampling: Bag;
  file_path: string;
  poster_path?: string;
  exists: boolean;
}

export interface DataDoc {
  projects: Project[];
  pipelines: Pipeline[];
  base_models: BaseModel[];
  dataset_snapshots: DatasetSnapshot[];
  experiments: Experiment[];
  checkpoints: Checkpoint[];
  validation_cases: ValidationCase[];
  output_videos: OutputVideo[];
  lineage: LineageEdge[];
}

export type AnnotationStatus = "interesting" | "candidate" | "broken" | null;

export interface Annotation {
  status: AnnotationStatus;
  notes: string;
  tags: string[];
}

export interface VideoSource {
  video: OutputVideo;
  experiment: Experiment;
  checkpoint: Checkpoint;
  validationCase: ValidationCase;
  baseModel?: BaseModel;
  dataset?: DatasetSnapshot;
  project?: Project;
  pipeline?: Pipeline;
}

export interface Column {
  key: string;
  experiment: Experiment;
  checkpoint: Checkpoint;
}
