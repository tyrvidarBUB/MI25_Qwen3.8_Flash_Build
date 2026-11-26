# Strata on two AMD Instinct MI25s

How to run Qwen3.8-Flash-Next IQ3_S on two MI25s. The steps below were run on one machine and are written so another PC with the same class of hardware can follow them.

Paths use `~/ai`. `~` is the home directory of the account that will run the server. Port **8083** leaves **8080** free for a llama.cpp server. The two servers cannot use the MI25s at the same time.

## Requirements

**64 GB of RAM is the minimum.** IQ3_S is Strata's 3.5-bit size. Its catalog entry says it needs a 64 GB PC with little else running (`ram_gb` 62, expert arena about 50 GB, download about 84 GB). Below 64 GB, pick a smaller size (IQ3_XXS, IQ2_XS, or Q2_0) or the model will page and stall.

This server configuration wants more than that floor. It keeps the expert set in RAM and also loads the 26.8 GiB n-gram table (`--ple-io ram`). Together with the KV cache and the operating system, plan on **96 GB or more**. The machine this was run on has **125 GB**. A 64 GB machine can still run IQ3_S if the n-gram table stays on disk: remove `--ple-io ram` from `strata-iq3_s.json` after setup.

| Item | This setup |
| --- | --- |
| RAM | 64 GB minimum. 96 GB or more for experts and the n-gram table together. Reference machine: 125 GB. |
| GPUs | Two AMD Instinct MI25, 16 GB HBM2 each (Vega 10, gfx900, PCI `1002:6861`). HIP numbers them GPU 0 and GPU 1. |
| Display | A separate display card. On the reference machine that is a Quadro P2000. It is not a Strata device. |
| CPU | AVX2. The reference CPU is a Xeon E5-2690 v3 (12 cores). The AVX-512 Q2_0 pack will not run on that CPU. IQ3_S will. |
| Disk | About 160 GB free. The model tree is about 133 GB after the model, pack, and MTP files. The ROCm venv is about 13 GB. The engine directory is about 800 MB. |
| OS | Linux with the `amdgpu` driver, `/dev/kfd`, and `/dev/dri`. Python 3.12, git, curl. These steps do not use Docker. |
| Port | 8083, bound to all IPv4 interfaces. An API key is required. |

Upstream Strata HIP packages do not include gfx900. gfx906 images do not run on these cards. The build below is the one that does.

## What gets installed

| Piece | Location |
| --- | --- |
| Source | `~/ai/strata`, `jonathanmajh/Strata` branch `gfx900-mi25`, commit `94517b9`, engine 0.1.59.1 |
| Python env | `~/ai/strata/.venv` |
| ROCm | TheRock wheels `rocm[libraries,devel]==7.14.0a20260612` from `https://rocm.nightlies.amd.com/v2/gfx900/` |
| llama.cpp | commit `3cf03257f219afbe7334045ff7c6a06ac68c627d` in `~/ai/strata/third_party/llama.cpp` |
| Engine | `~/ai/strata/engine/strata` and the gfx900 ROCm libraries beside it |
| Weights | `~/ai/Strata-data/models/IQ3_S/` |
| Pack and MTP | `~/ai/Strata-data/packs/iq3_s` and `~/ai/Strata-data/mtp/rt` |
| Run config | `~/ai/strata/strata-iq3_s.json` |
| Launcher | `~/ai/run-strata.sh` |
| API key | `~/ai/strata-api-key` (mode 600). Generate a new one on each machine. Do not commit it or send it with this document. |
| Log | `~/ai/strata/strata-iq3_s.log` |

## 1. Let your user open the GPUs

Both MI25s must show up as KFD GPU nodes, and your user must be able to open `/dev/kfd` and the render nodes under `/dev/dri`.

```bash
python3 -c 'open("/dev/kfd","rb").close(); print("kfd ok")'
```

If that fails, add a seat ACL and the render group. A udev rule that does this:

```
SUBSYSTEM=="kfd", KERNEL=="kfd", TAG+="uaccess", GROUP="render", MODE="0660"
```

Save that as `/etc/udev/rules.d/70-kfd-uaccess.rules`, then:

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
sudo usermod -aG render,video "$USER"
```

Group membership applies to a new login. An ACL on `/dev/kfd` covers the current session. Confirm both cards before building:

```bash
python3 - << 'PY'
import os
base = "/sys/class/kfd/kfd/topology/nodes"
n = 0
for name in sorted(os.listdir(base), key=int):
    props = {}
    for line in open(os.path.join(base, name, "properties")):
        k, _, v = line.partition(" ")
        props[k] = v.strip()
    if int(props.get("simd_count") or 0) == 0:
        continue
    print(f"HIP {n}: node {name} name={open(os.path.join(base, name, 'name')).read().strip()}")
    n += 1
PY
```

HIP 0 and HIP 1 must both be `vega10`. The display card does not appear in this list.

## 2. Clone the gfx900 branch

```bash
git clone --depth 1 --branch gfx900-mi25 https://github.com/jonathanmajh/Strata.git ~/ai/strata
cd ~/ai/strata
python3 -m venv .venv
.venv/bin/pip install --disable-pip-version-check -r requirements-build.txt patchelf
```

That installs CMake 4.4.3, Ninja, and patchelf into the venv. Put `.venv/bin` first on `PATH` for the build.

`setup.py` on this branch will compile the wave32 HIP backend if you let it build the engine itself. That backend is fatal next to gfx900. Before any `./setup.sh --backend hip`, `build_engine_hip()` in `setup.py` must pass `-DSTRATA_HIP_GFX900=ON` and `-DSTRATA_ENABLE_HIP=OFF`, and it must leave `-DSTRATA_PREFILL_MMQ` off. The engine used here was produced by `packaging/build_gfx900_linux.sh`, then copied into `engine/` with a `BUILD.json` so setup sees it as already built.

## 3. Install the gfx900 ROCm wheels

```bash
cd ~/ai/strata
.venv/bin/pip install --disable-pip-version-check \
  --index-url https://rocm.nightlies.amd.com/v2/gfx900/ \
  'rocm[libraries,devel]==7.14.0a20260612'
```

This is about 3 GB. When it finishes:

```bash
.venv/bin/rocminfo | awk '/Agent |Marketing Name:|Device Type: GPU/'
```

Expect two `AMD Radeon Pro WX 9100` GPU agents. That is the flashed name of these MI25s.

## 4. Fetch the pinned llama.cpp

The engine links the checkout's pinned llama.cpp, commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`:

```bash
cd ~/ai/strata
.venv/bin/python -c 'import setup; setup.get_llama_cpp()'
test -f third_party/llama.cpp/ggml/CMakeLists.txt && echo llama-ok
```

## 5. Build the gfx900 engine

Resolve the wheel roots, then run the packaging script. Set `JOBS` to the number of physical cores. The reference CPU has 12, so the command below uses `JOBS=12`. `VISION_CPU=0` skips the picture reader. Pictures are off in the server config.

```bash
cd ~/ai/strata
export PATH="$PWD/.venv/bin:$PATH"
HP=$(.venv/bin/rocm-sdk path --root | head -n1)
CORE=$(find .venv -type d -name _rocm_sdk_core | head -n1)
LIBW=$(find .venv -type d -name '_rocm_sdk_libraries_*' | head -n1)
VARIANT=gfx900 ARCHS=gfx900 SRC="$PWD" \
  LLAMA="$PWD/third_party/llama.cpp" \
  HP="$HP" CORE="$CORE" LIBW="$LIBW" \
  OUT="$PWD/work-gfx900" PKGVER=0.1.59.1 \
  COMMIT=$(git rev-parse HEAD) JOBS=12 VISION_CPU=0 \
  packaging/build_gfx900_linux.sh
```

CMake must report `HIP (gfx906 compat) enabled, arch gfx900`. On an AVX2-only CPU the CPU backend is AVX2.

Install the staged binary where setup looks for it:

```bash
rm -rf engine
mkdir -p engine
cp -a work-gfx900/stage/. engine/
```

Write `engine/BUILD.json` with `"backend": "hip"`, `"version": "0.1.59.1"`, `"archs": ["gfx900"]`, `"vision": "none"`, `"lib_dirs"` set to the absolute path of that `engine` directory, and `"src"` set to the fingerprint setup uses (`setup.source_hash(setup.ENGINE_SOURCES)`). A matching stamp makes a later setup skip the compile. The binary's runpath is `$ORIGIN`, so the ROCm libraries stay in `engine/` next to `strata`.

Check the version string:

```bash
.venv/bin/python -c 'import pathlib,re; b=pathlib.Path("engine/strata").read_bytes(); print(re.search(br"engine=([0-9.]+)", b).group(1).decode())'
```

It prints `0.1.59.1`.

## 6. Download IQ3_S

Weights go in `~/ai/Strata-data/models/IQ3_S/`. Both files are from ISTA-DASLab revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`. Shard 2 is the shared n-gram table.

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` | 54817524224 | `4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3` |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` | 28800138432 | `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |

```bash
DEST=~/ai/Strata-data/models/IQ3_S
BASE=https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ3_S
mkdir -p "$DEST"
curl -L --fail -C - -o "$DEST/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf" \
  "$BASE/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"
curl -L --fail -C - -o "$DEST/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf" \
  "$BASE/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf"
```

`-C -` resumes a partial file. Setup hashes both files against the table above and refuses a short or different file.

Leave the downloads alone while they run. On one home link, a second large download (the ROCm wheels, or the 5 GB MTP fetch) will starve them.

## 7. Prepare the model and write the server config

Generate an API key first. Each machine gets its own key:

```bash
umask 077
openssl rand -hex 24 > ~/ai/strata-api-key
chmod 600 ~/ai/strata-api-key
```

Then, from `~/ai/strata`, with the gfx900 engine already in `engine/`:

```bash
export STRATA_EXPERIMENTAL_GFX900=1
export PYTHONUNBUFFERED=1
.venv/bin/python setup.py \
  --yes \
  --backend hip \
  --family qwen \
  --model IQ3_S \
  --context 262144 \
  --kv int8 \
  --vision no \
  --gpus 0,1 \
  --layer-split 27 \
  --port 8083 \
  --host 0.0.0.0 \
  --api-key "$(cat ~/ai/strata-api-key)" \
  --no-browser \
  --no-start \
  --mtp on \
  --experimental-speed-projection off
```

What that does:

- Accepts both MI25s only when `STRATA_EXPERIMENTAL_GFX900=1` is set. `--backend hip` keeps a non-AMD display card out of the choice.
- Checks the two GGUF files, builds `packs/iq3_s`, and fetches about 5 GB of MTP tensors into `~/ai/Strata-data/mtp`.
- Writes `strata-iq3_s.json` and `run-iq3_s.sh`.
- Does not start the server (`--no-start`).

`--layer-split 27` is the first layer of the second GPU, not a layer count per card. The running engine uses layers 0–26 on HIP 0 and layers 27–47 on HIP 1.

Setup also records the rocBLAS table directory. The file that matches this wheel is `tools/hip/gfx900-rocblas-5.5.0.9811f2ea52.txt`. A missing or mismatched table is a warning. The engine then uses its default kernels. gfx900 has no hipBLASLt table. That warning is expected.

## 8. Server configuration

`~/ai/strata/strata-iq3_s.json` is the saved server:

| Setting | Value |
| --- | --- |
| Model | `qwen3.8-flash-next-iq3_s` |
| GPUs | `[0, 1]`, layer split `27` |
| Context | 262144 tokens, KV `int8` |
| KV resident on GPU | 32768 cells per layer. The rest of the 262144 window streams from RAM (about 3.6 GB). |
| Experts | Native IQ3_S pack. Cache `auto` on each card. |
| N-gram table | `--ple-io ram` (26.8 GiB in RAM) |
| MTP | On. Draft runtime in `~/ai/Strata-data/mtp/rt`. Speculative depth 4, min probability 0.5. |
| Pictures | Off |
| Bind | `0.0.0.0:8083` |
| API key | The value in the config, same as `~/ai/strata-api-key` |
| Browser | Does not open on start |

`~/ai/run-strata.sh` is the launcher. It exports the gfx900 runtime switches and then execs `run-iq3_s.sh`:

| Variable | Role |
| --- | --- |
| `STRATA_EXPERIMENTAL_GFX900=1` | Lets the card check accept Vega 10 |
| `STRATA_ROCBLAS_TUNING` | Path of `~/ai/strata/tools/hip`. Loads the gfx900 rocBLAS table when the library version matches. |
| `STRATA_SPLIT_OVERLAP=1` | Overlaps the hand-off between the two cards |
| `STRATA_STAGE_TRIM=1` | Each card loads only its own layers. Requires the explicit split of 27. |
| `STRATA_ARENA_MMAP=1` | Map the expert arena instead of a second copy |
| `STRATA_PREFILL_CPU_SHARE=auto` | CPU share of prefill, measured on the fork author's box |
| `STRATA_HIP_PROMPT_F16=1` | FP16 prompt GEMMs. Output bytes can differ from the default path. |
| `STRATA_GFX906_ATTN_QUERY_SWIZZLE=1` | Vega attention path |
| `STRATA_GFX906_ATTN_REDUCE12=1` | Vega attention path |

The config's own `env` also sets `STRATA_ROCBLAS_TUNING`. The launcher sets the rest for the process.

## 9. Start, stop, and check

Stop any other server that is using the MI25s first. A local Ollama that is not holding these cards can stay.

```bash
~/ai/run-strata.sh
```

Leave that terminal open. Ctrl-C stops the server. The first start reads about 55 GB of experts into RAM and fills both cards. That takes a minute or two. The window prints `ready` and the URLs when the session is up.

To leave it running after the terminal closes:

```bash
~/ai/run-strata.sh > ~/ai/strata/strata-serve.out 2>&1 &
echo $! > ~/ai/strata/strata.pid
```

Stop that process with:

```bash
kill "$(cat ~/ai/strata/strata.pid)"
```

Chat and API:

- This PC: `http://127.0.0.1:8083`
- Another PC on the LAN: `http://<machine-name>.local:8083`, or `http://<lan-address>:8083`
- OpenAI: `http://127.0.0.1:8083/v1/chat/completions`
- Model name: `qwen3.8-flash-next-iq3_s`

If a firewall is enabled, allow TCP 8083 before the LAN URL will answer.

A client must send the API key (`Authorization: Bearer` and the contents of `~/ai/strata-api-key`). Qwen thinking is on by default, so a short `max_tokens` can be spent entirely on `reasoning_content` and the visible `content` stays empty. Send `"reasoning_effort": "none"` when you want the answer text directly.

Both cards should be nearly full after load (about 15.4 GiB of 16 GiB each) and both should go busy during a reply. The engine log names them `AMD Radeon Pro WX 9100` and `gfx900:xnack-`, and it prints `layer split: layers 0-26 (CUDA0), 27-47 (CUDA1)`. The CUDA names are the HIP compatibility layer.

## Limits

- **RAM.** 64 GB is the minimum for IQ3_S. This config, with the n-gram table in RAM, is sized for a machine with about 96 GB or more. The reference machine has 125 GB.
- **Locked memory.** Check `ulimit -Hl`. On the reference machine the hard limit is 15.7 GiB, so the 26.8 GiB n-gram table is faulted into RAM and not fully pinned. The server still answers. Raising the limit for the service (which needs a higher hard limit) was measured at up to about 6% faster decode.
- **One pair of cards.** Strata holds both MI25s. Anything else that needs those cards, including a llama.cpp Vulkan server, has to be stopped first.
- **Do not rebuild with the stock HIP flags.** `-DSTRATA_ENABLE_HIP=ON` and `-DSTRATA_PREFILL_MMQ=ON` are the wave32 path. gfx900 uses `-DSTRATA_HIP_GFX900=ON` only.
