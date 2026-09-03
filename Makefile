# One-command reproduction. Each target skips work that is already on disk, so an interrupted
# run continues rather than restarting.
#
# Before anything: accept the two licences while signed in to Hugging Face, then `hf auth login`.
#   https://huggingface.co/CohereLabs/tiny-aya-global          (CC-BY-NC)
#   https://huggingface.co/datasets/openlanguagedata/flores_plus (CC-BY-SA 4.0)

PY := .venv/bin/python
SH := .venv/bin/bash

.PHONY: all setup check arms data weights bpb speed translate quality figures reproduce clean-figures

all: reproduce

setup:
	uv venv --python 3.12
	uv pip install mlx-lm huggingface_hub sacrebleu datasets pandas matplotlib scipy \
	               statsmodels lingua-language-detector transformers

check:                       ## architecture support and tied embeddings
	$(PY) convert/gate_check.py
	$(PY) convert/check_tie.py
	$(PY) analysis/param_budget.py

arms: models/E-q4-emb8       ## the five quantization arms
models/E-q4-emb8:
	bash convert/make_arms.sh
	$(PY) convert/arm_e.py
	$(PY) convert/sanity_check.py

data: data/lang_meta.csv     ## FLORES+ parallel corpus and tokenizer fertility
data/lang_meta.csv:
	$(PY) eval/prepare_data.py

weights: results/token_error.csv   ## L0, needs no inference
results/token_error.csv: data/lang_meta.csv
	$(PY) analysis/weight_error.py

bpb: results/bpb.csv         ## L1, about 3 hours, resumable per arm and language
results/bpb.csv: models/E-q4-emb8 data/lang_meta.csv
	$(PY) eval/bpb.py

speed: results/speed.csv     ## L5, about 90 minutes, needs an otherwise idle machine
results/speed.csv: models/E-q4-emb8 data/lang_meta.csv
	$(PY) bench/make_order.py
	$(PY) bench/speed.py

translate: outputs/E-q4-emb8_mya_Mymr.jsonl   ## L2 generation, about 8 hours
outputs/E-q4-emb8_mya_Mymr.jsonl: models/E-q4-emb8 data/lang_meta.csv
	$(PY) eval/translate.py

quality: results/chrf.csv results/fidelity.csv   ## L2 scoring and L3 fidelity
results/chrf.csv results/fidelity.csv: outputs/E-q4-emb8_mya_Mymr.jsonl
	$(PY) eval/score_chrf.py
	$(PY) eval/fidelity.py

stats: results/bpb.csv results/chrf.csv          ## the registered tests
	$(PY) analysis/stats.py    | tee results/stats_bpb.txt
	$(PY) analysis/stats_l2.py | tee results/stats_chrf.txt

figures:
	$(PY) analysis/plot_fertility.py
	$(PY) analysis/plot_weight_error.py
	$(PY) analysis/plot_results.py

# Everything except the two long generation jobs. Roughly four hours on an M3 Air.
reproduce: check arms data weights bpb figures
	@echo "core pipeline complete. run 'make translate quality stats' for the generation results."

clean-figures:
	rm -f figures/0*.png
