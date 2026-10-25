# MMMC layer-0 conflict detection
Goal: download official MMMC into /data/datasets; evaluate InternVL, Qwen, MoF, DeepSeek using transformer block index 0; isolate all new artifacts here.

## Phases
1. complete: inspect official dataset schema, existing model/evidence implementations and environments.
2. complete (test): downloaded and SHA256-validated official 4000-row test; created genuine paired pilot. Full train download and requested /data/datasets location remain pending.
3. in_progress: InternVL isolated layer0 runner and two-row GPU smoke passed; Qwen/MoF/DeepSeek pending per priority.
4. complete (InternVL pilot): 200 source-matched rows, AUROC0.4398/AUPRC0.4654 with paired bootstrapCI; report and figures generated. Remaining models pending.

## Constraints
Do not modify prior results or existing experiment code. No training requested. Layer 0 means first decoder block, not embedding output.

## User steering
先做 InternVL；仅使用现有 internvl 环境，禁止修改环境或安装/升级/卸载包。其余三个模型后续再开展。

5. complete: User requested final-layer comparison on the identical InternVL100-pair pilot; preserve layer0 outputs and existing environment.
