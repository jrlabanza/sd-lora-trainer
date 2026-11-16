# sd-forge-lora-trainer

Adds a **LoRA Trainer** tab to Stable Diffusion WebUI Forge - Neo.

## What's new - November 2026: Auto-LoRA

A **🧬 Auto-LoRA** panel at the top of the tab goes from one picture to a trained character LoRA:

1. Drop a **character picture** and press **🔎 Identify character**: the WD14 tagger reads it and fills in the Danbooru character tag (you can type or correct it, e.g. `raiden mei`).
2. Set a **trigger word** (it becomes the first tag of every caption), the **dataset rating** (safe / sensitive / questionable / explicit / all) and the **dataset size** (20-120 images, 48 by default).
3. Press **Train**: it collects a booru dataset for that character, captions it with WD14, trains with the Character preset on your base checkpoint, and - when ticked - registers the result in `styles.csv` and saves a **Passport** for it. Progress streams into the *Auto-LoRA job log*.

It stops with a message when too few usable images are found for a clean LoRA or no base checkpoint is installed. The manual controls below stay available for full control.

## What it does

Drop 5–15 reference images, type a trigger word, pick a preset (Character / Style / Concept) and a base SDXL checkpoint, then click **Start training**. The extension automatically:

1. Installs `kohya-ss/sd-scripts` into a sandboxed venv at `extensions/sd-forge-lora-trainer/sd-scripts-venv/` on first use (so Forge's bleeding-edge PyTorch doesn't get downgraded).
2. Resizes / pads your images to SDXL bucket sizes.
3. Auto-captions with WD14 tagger (anime) or BLIP (photo), then prepends your trigger word to every caption.
4. Writes the kohya folder layout (`N_concept/...`) and a training TOML.
5. Launches `sdxl_train_network.py` as a subprocess inside the sandboxed venv, streaming stdout to the UI log.
6. Copies the resulting `<project>.safetensors` into `models/Lora/` so it's immediately usable as `<lora:<project>:1>`.

## Defaults

Tuned for 8–12 GB cards:
- `network_dim` 8–32 (preset-dependent), `network_alpha` half of dim
- gradient checkpointing on, fp16 mixed precision, no_half_vae on
- `cache_text_encoder_outputs` to disk (huge VRAM saver on SDXL)
- 8-bit AdamW optimizer (bitsandbytes), `min_snr_gamma=5`
- multires noise + noise offset for richer LoRAs

Override any of these in the **Advanced** accordion.

## First-time install

The first `Start training` click downloads ~5–7 GB of dependencies (PyTorch CUDA 12.1 + accelerate + transformers + diffusers + bitsandbytes + onnxruntime-gpu) into the sandboxed venv. Expect 5–15 minutes on a fast connection. After that, subsequent training jobs start in seconds.

If your driver doesn't support CUDA 12.x wheels (rare), set this in `webui-user.bat` before launching once:

```bat
set LORA_TRAINER_TORCH_INDEX=https://download.pytorch.org/whl/cu118
```

## Troubleshooting

- **OOM during training** → drop `network_dim` to 8, set `train_batch_size` to 1, lower resolution to 768.
- **NaN loss / black previews** → make sure `no_half_vae` stayed on; this is the same NaN class as the IP-Adapter black-image bug.
- **WD14 download fails** → kohya pulls the WD14 ONNX from HuggingFace on first use. Check your network / huggingface-hub token if behind a proxy.

## Files written

```
extensions/sd-forge-lora-trainer/
├── install.py                  # minimal startup stub
├── scripts/lora_trainer.py     # the tab + orchestration
├── sd-scripts/                 # kohya repo, cloned on first use
├── sd-scripts-venv/            # sandboxed venv, created on first use
└── projects/<project_name>/    # per-project working dir
    ├── images/<N>_<class>/     # resized inputs + .txt captions
    ├── config.toml             # generated kohya config
    ├── out/                    # training checkpoints
    └── logs/                   # tensorboard logs
```
