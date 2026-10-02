# Agentomics PHACT models

Retained model implementation and input-representation code:

- `model5/model_training/train.py`: single-candidate PHACT multimodal model and training.
- `model5/model_inference/inference.py`: single-candidate inference.
- `model4/layer_mix_training/`: PHACT-conditioned RiNALMo layer mixing and training.
- `model4/layer_mix_inference/inference.py`: layer-mix inference.
- `model4/multi_candidate/`: multi-candidate representation, model, training and inference.

Use `model5/environment.yml` for the recorded runtime. Training and inference
entry points accept explicit input, output and artifact paths; use `--help`
for their arguments. Required data, pretrained components and trained weights
are supplied outside this repository. The existing model source and its
relative directory relationships are preserved.
