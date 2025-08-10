import { z } from 'zod'

// Base schemas
export const TargetSchema = z.object({
  target_id: z.string(),
  target_entry: z.string(),
  sequence: z.string(),
  n_train: z.number(),
  n_val: z.number(),
  n_test: z.number(),
  pos_frac_train: z.number(),
})

export const HealthSchema = z.object({
  status: z.literal('ok'),
  model: z.enum(['fusion', 'baseline']),
  calibrated: z.boolean(),
  commit: z.string(),
})

export const PredictRequestSchema = z.object({
  target_id: z.string(),
  smiles: z.string(),
  seed: z.number().optional().default(42),
  calibrate: z.boolean().optional().default(true),
  enable_ood_check: z.boolean().optional().default(true),
})

export const PredictResponseSchema = z.object({
  proba: z.number(),
  pkd: z.number().optional(),
  latency_ms: z.number(),
  model: z.enum(['fusion', 'baseline']),
  calibrated: z.boolean(),
  temperature: z.number().optional(),
  abstained: z.boolean(),
  ood: z.boolean(),
})

export const BatchRequestSchema = z.object({
  rows: z.array(z.object({
    target_id: z.string(),
    smiles: z.string(),
  })),
  calibrate: z.boolean().optional().default(true),
  abstain_band: z.tuple([z.number(), z.number()]).optional().default([0.45, 0.55]),
  threads: z.number().optional().default(4),
})

export const BatchResultSchema = z.object({
  idx: z.number(),
  target_id: z.string(),
  smiles: z.string(),
  proba: z.number(),
  pkd: z.number().optional(),
  latency_ms: z.number(),
  abstained: z.boolean(),
  ood: z.boolean(),
})

export const BatchResponseSchema = z.object({
  results: z.array(BatchResultSchema),
})

export const ExplainRequestSchema = z.object({
  target_id: z.string(),
  smiles: z.string(),
})

export const AtomImportanceSchema = z.object({
  atom_idx: z.number(),
  score: z.number(),
})

export const ResidueWindowSchema = z.object({
  start: z.number(),
  end: z.number(),
  score: z.number(),
})

export const FragmentSchema = z.object({
  bit: z.number(),
  smarts: z.string(),
  weight: z.number(),
})

// Scientific Analysis Schemas
export const MolecularPropertiesSchema = z.object({
  molecular_weight: z.number(),
  logp: z.number(),
  hbd: z.number(),
  hba: z.number(),
  tpsa: z.number(),
  rotatable_bonds: z.number(),
  aromatic_rings: z.number(),
  lipinski_violations: z.number(),
  drug_likeness_score: z.number(),
})

export const BindingAnalysisSchema = z.object({
  binding_affinity_class: z.string(),
  confidence_level: z.string(),
  key_interactions: z.array(z.string()),
  binding_mode: z.string(),
  selectivity_profile: z.record(z.number()),
})

export const ExplainResponseSchema = z.object({
  molecular_properties: MolecularPropertiesSchema,
  binding_analysis: BindingAnalysisSchema,
  structural_alerts: z.array(z.string()),
  optimization_suggestions: z.array(z.string()),
  chemical_novelty_analysis: z.array(z.string()),
  confidence_score: z.number(),
})

export const PerTargetMetricSchema = z.object({
  target_id: z.string(),
  auroc: z.number(),
  prauc: z.number(),
  ece_after: z.number(),
  n_test: z.number(),
})

export const MetricsResponseSchema = z.object({
  global: z.object({
    auroc: z.number(),
    prauc: z.number(),
    ece_before: z.number(),
    ece_after: z.number(),
    lat_p50_ms: z.number(),
    lat_p95_ms: z.number(),
    model_size_mb: z.number(),
  }),
  per_target: z.array(PerTargetMetricSchema),
  plots: z.object({
    roc_png: z.string(),
    pr_png: z.string(),
    reliability_png: z.string(),
  }),
  split: z.string(),
})

// Inferred types
export type Target = z.infer<typeof TargetSchema>
export type Health = z.infer<typeof HealthSchema>
export type PredictRequest = z.infer<typeof PredictRequestSchema>
export type PredictResponse = z.infer<typeof PredictResponseSchema>
export type BatchRequest = z.infer<typeof BatchRequestSchema>
export type BatchResult = z.infer<typeof BatchResultSchema>
export type BatchResponse = z.infer<typeof BatchResponseSchema>
export type ExplainRequest = z.infer<typeof ExplainRequestSchema>
export type MolecularProperties = z.infer<typeof MolecularPropertiesSchema>
export type BindingAnalysis = z.infer<typeof BindingAnalysisSchema>
export type ExplainResponse = z.infer<typeof ExplainResponseSchema>
export type PerTargetMetric = z.infer<typeof PerTargetMetricSchema>
export type MetricsResponse = z.infer<typeof MetricsResponseSchema>

// UI-specific schemas
export const ThresholdPresetSchema = z.object({
  max_f1: z.number(),
  precision_90: z.number(),
  recall_90: z.number(),
})

export const ConfusionMatrixSchema = z.object({
  tp: z.number(),
  fp: z.number(),
  fn: z.number(),
  tn: z.number(),
})

export type ThresholdPreset = z.infer<typeof ThresholdPresetSchema>
export type ConfusionMatrix = z.infer<typeof ConfusionMatrixSchema>
