# Codex Task: B17 — WG_SIZE 64→256 nel monolitico V3D
# Project: ADAM — Vulkan LLM inference, Gemma3-1B, Pi 5 V3D
# Date: 2026-04-04

---

## Root cause identificato

Su V3D (Pi 5) il bottleneck del monolitico non sono i barrier né la bandwidth DRAM —
è la **latenza per cache line ARM** moltiplicata per il numero di transazioni seriali.

Dati misurati:
- `map_matvec_t_xq8` (shader dedicato) per 1 matvec N=6912, K=1152: **311ms**
- Monolitico completo 26 layer: **530ms**
- Bandwidth effettiva SSBO: ~25 MB/s su 8+ GB/s disponibili

Causa: V3D ha 16 QPU hardware. Con WG_SIZE=64 il monolitico usa 4 wave da 16 thread.
Con N_WG=1 (forzato su V3D), 12 QPU restano idle durante i memory access.
Ogni cache line (64 byte) viene fetchata serialmente: 124k transazioni × 2.5µs = 311ms/matvec.

Con WG_SIZE=256: 16 wave da 16 thread → tutti e 16 i QPU saturati → pipeline di
prefetch hardware 4× più profonda → ~4× riduzione latenza effettiva.

V3D supporta fino a 256 invocazioni per workgroup (verificato da maxComputeWorkGroupSize).

---

## Modifica: solo `map_full_decode_step.comp`, solo path ADAMAH_V3D_MONOLITHIC

### 1. WG_SIZE da 64 a 256 su V3D

```glsl
// PRIMA:
#if defined(ADAMAH_PROFILE_BROADCOM_V3DV_BALANCED) || defined(ADAMAH_PROFILE_BROADCOM_V3DV_NARROW)
#define ADAMAH_V3D_MONOLITHIC 1
#else
#define ADAMAH_V3D_MONOLITHIC 0
#endif

#if ADAMAH_V3D_MONOLITHIC
#define WG_SIZE 64
#else
#define WG_SIZE 128
#endif

// DOPO:
#if defined(ADAMAH_PROFILE_BROADCOM_V3DV_BALANCED) || defined(ADAMAH_PROFILE_BROADCOM_V3DV_NARROW)
#define ADAMAH_V3D_MONOLITHIC 1
#else
#define ADAMAH_V3D_MONOLITHIC 0
#endif

#if ADAMAH_V3D_MONOLITHIC
#define WG_SIZE 256
#else
#define WG_SIZE 128
#endif
```

### 2. sh_reduce dimensionato su WG_SIZE (già corretto — nessuna modifica)

`sh_reduce[WG_SIZE]` usa la macro, si adatta automaticamente.

### 3. Verificare che MAX_TMP e MAX_EMBD reggano WG_SIZE=256

Le shared arrays `sh_hidden[MAX_EMBD]` e `sh_tmp[MAX_TMP]` NON dipendono da WG_SIZE —
sono dimensionate sui dati del modello (MAX_EMBD=2048, MAX_TMP=9216), non sul numero
di thread. Nessuna modifica necessaria.

### 4. Loop thread-per-row (già B16) — nessuna modifica

Con WG_SIZE=256 e thread-per-row, ogni thread fa N/256 righe invece di N/64.
Il pattern è già corretto dopo B16.

### 5. Loop reduce_sum / reduce_max in rmsnorm e attention

Questi usano `sh_reduce[WG_SIZE]` con tree reduction `for (s = WG_SIZE>>1; s > 0; s >>= 1)`.
Con WG_SIZE=256: 8 iterazioni invece di 6. Corretto automaticamente — nessuna modifica.

---

## Cosa NON modificare

- `adamah.c` — nessuna modifica
- Python engine — nessuna modifica
- Nessun altro shader
- Non aumentare `dispatch_wg` in `adamah_full_decode_step` — rimane N_WG=1 su V3D

---

## Build sul Pi

```bash
cd ~/ADAM/adamah-MAIN/adamah
glslc -DADAMAH_PROFILE_BROADCOM_V3DV_BALANCED \
  shaders/src/f32/map_full_decode_step.comp \
  -o shaders/f32/map_full_decode_step.spv
cp shaders/f32/map_full_decode_step.spv shaders/map_full_decode_step.spv

gcc -shared -O2 -march=native \
  -include _shader_path.h \
  -I"$VULKAN_SDK/include" \
  adamah.c -o adamah.so \
  -lvulkan -lm
# Verifica: ~220KB
```

---

## Validazione

```bash
# Correttezza prima di tutto
PYTHONUTF8=1 PYTHONPATH=~/ADAM python3 -X utf8 \
  tests/diagnostics/diag_inference.py gemma3-1b.gguf
# Must be: 8/8 PASS, "2+2=4"

# Performance
PYTHONUTF8=1 PYTHONPATH=~/ADAM python3 -X utf8 \
  tests/diagnostics/diag_chat_perf.py gemma3-1b.gguf --max-tokens 8
# Baseline: Turn2 core_batch ~530ms, decode_tps ~1.24
# Target:   Turn2 core_batch <200ms, decode_tps >4.0
# Ottimo:   Turn2 core_batch ~130ms, decode_tps ~7.5
```

---

## Se non migliora

Se core_batch rimane ~530ms con WG_SIZE=256, significa che V3D non parallelizza
le wave aggiuntive per accessi SSBO — il limite è il memory controller, non il
numero di QPU attivi. In quel caso riportare il risultato in AGENT_COLLAB.md
e non esplorare ulteriori variazioni di WG_SIZE.

---

## Post results in AGENT_COLLAB.md

- diag_inference Pi: PASS count
- core_batch_ms prima e dopo
- decode_tps Turn2
- Se regredisce: ripristinare WG_SIZE=64 immediatamente
