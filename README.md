# sd-forge-lora-trainer

Adds a **LoRA Trainer** tab to Stable Diffusion WebUI Forge - Neo.

## What's new - November 2026: setup for every card, done by Forge's installer

- **RTX 50 cards train.** The training environment gets PyTorch 2.8.0 + CUDA 12.8 on an RTX 50 (Blackwell, as
  sd-scripts' README asks) - the 2.6.0 + CUDA 12.4 it always installed has no kernels for those cards. Other NVIDIA
  cards keep 2.6.0 + CUDA 12.4, the version sd-scripts is tested with.
- **Set up by Forge's installer.** `tools\initialize.bat` (and the AI Studio Hub installer) builds the training
  environment with a private Python 3.11 - no double-click needed (`-NoTrainer` skips it; `setup_venv.bat` still does
  it by hand). Re-running is quick: a working environment is only checked.
- **Pinned and checked.** kohya-ss/sd-scripts is cloned at the commit this extension was tested with (`068bcd7`), and
  setup ends with a calculation on the GPU and the training stack's imports - exit code 0 means ready.

## Auto-LoRA

A **🧬 Auto-LoRA** panel at the top of the tab goes from one picture to a trained character LoRA:

1. Drop a **character picture** and press **🔎 Identify character**: the WD14 tagger reads it and fills in the Danbooru character tag (you can type or correct it, e.g. `raiden mei`).
2. Set a **trigger word** (it becomes the first tag of every caption), the **dataset rating** (safe / sensitive / questionable / explicit / all) and the **dataset size** (20-120 images, 48 by default).
3. Press **Train**: it collects a booru dataset for that character, captions it with WD14, trains with the Character preset on your base checkpoint, and - when ticked - registers the result in `styles.csv` and saves a **Passport** for it. Progress streams into the *Auto-LoRA job log*.

It stops with a message when too few usable images are found for a clean LoRA or no base checkpoint is installed. The manual controls below stay available for full control.

## What it does

Drop 5–15 reference images, type a trigger word, pick a preset (Character / Style / Concept) and a base SDXL checkpoint, then click **Start training**. The extension automatically:

1. Runs `kohya-ss/sd-scripts` from its own sandboxed venv at `extensions/sd-forge-lora-trainer/sd-scripts-venv/` (so Forge's bleeding-edge PyTorch doesn't get downgraded) - set up once, see below.
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

Forge's installer (`tools\initialize.bat`, step *LoRA Trainer environment*) does it on an NVIDIA card; by hand,
double-click `setup_venv.bat` (Python 3.10 or 3.11 via the `py` launcher). `setup_bootstrap.py` clones sd-scripts at
`068bcd7`, creates `sd-scripts-venv/` and installs PyTorch for the card (2.8.0 + CUDA 12.8 on an RTX 50, NVIDIA driver
570+; else 2.6.0 + CUDA 12.4), kohya's `requirements.txt` and onnxruntime-gpu - about 8 GB, 5–15 minutes - then checks
it with a calculation on the GPU. Run again, it only checks (and repairs what is missing).

To force another PyTorch, set before running it:

```bat
set LORA_TRAINER_TORCH_INDEX=https://download.pytorch.org/whl/cu118
set LORA_TRAINER_TORCH=torch==2.6.0 torchvision==0.21.0
```

Linux: the tab is installed by `linux/initialize.sh`, but its training environment is not set up there yet (the
container has only Forge's Python 3.13).

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
