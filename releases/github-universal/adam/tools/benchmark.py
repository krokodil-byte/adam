"""ADAM vs Ollama — GGUF Model Benchmark"""
import os, sys, json, time, argparse
import urllib.request, urllib.error

# Ensure the project root is in sys.path so 'adam' and 'adamah' are importable
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ADAMAH_DIR = os.path.join(_ROOT, "adamah-MAIN")
for _p in [_ROOT, _ADAMAH_DIR]:
    if _p not in sys.path: sys.path.insert(0, _p)

class ADAMRunner:
    def __init__(self, model_path):
        self.model_path = model_path
        self.engine = self.tokenizer = self.config = None

    def load(self, verbose=True):
        from adam.loaders.gguf import GGUFLoader
        from adam.tokenizers.gguf_tok import GGUFTokenizer
        from adam.models.engine import ADAMEngine, ModelConfig, GenerationConfig
        import adamah as adamah_mod

        loader = GGUFLoader(self.model_path); loader.load(verbose=verbose)
        self.tokenizer = GGUFTokenizer(
            vocab=loader.get_tokenizer_vocab(), scores=loader.get_tokenizer_scores(),
            bos_id=loader.get_bos_token_id(), eos_id=loader.get_eos_token_id())

        cfg = ModelConfig.from_gguf_metadata(loader.metadata, verbose=verbose)

        if verbose: print("[ADAM] Init GPU...")
        gpu = adamah_mod.init()
        self.engine = ADAMEngine(gpu, cfg, loader.tensors,
                                 raw_blocks=loader.raw_blocks,
                                 tensor_types=loader.tensor_types,
                                 adamah_mod=adamah_mod, verbose=verbose)
        self.config = cfg
        self._GenConfig = GenerationConfig
        if verbose: print("[ADAM] Ready!")
        return self

    def run(self, prompt, max_tokens=64):
        tokens = self.tokenizer.encode(prompt)
        cfg = self._GenConfig(max_tokens=max_tokens, temperature=0.7,
                              top_k=40, top_p=0.95, seed=42)
        out, stats = self.engine.generate(tokens, cfg)
        stats['text'] = self.tokenizer.decode(out)
        return stats

class OllamaRunner:
    def __init__(self, model, url='http://localhost:11434'):
        self.model = model; self.url = url

    def check(self):
        try:
            with urllib.request.urlopen(f'{self.url}/api/tags', timeout=5) as r:
                ms = [m['name'] for m in json.loads(r.read()).get('models', [])]
                return any(self.model in m for m in ms)
        except: return False

    def run(self, prompt, max_tokens=64):
        body = json.dumps({'model': self.model, 'prompt': prompt, 'stream': False,
            'options': {'num_predict': max_tokens, 'temperature': 0.7,
                        'top_k': 40, 'top_p': 0.95, 'seed': 42}}).encode()
        req = urllib.request.Request(f'{self.url}/api/generate', data=body,
                                      headers={'Content-Type': 'application/json'})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.loads(r.read())
        wall = time.perf_counter() - t0
        nt = d.get('eval_count', 0); dur = d.get('eval_duration', 0)
        pc = d.get('prompt_eval_count', 0); pd = d.get('prompt_eval_duration', 0)
        return {'text': d.get('response', ''), 'n_gen': nt, 'n_prompt': pc,
                'decode_tps': nt / (dur/1e9) if dur > 0 else 0,
                'prefill_tps': pc / (pd/1e9) if pd > 0 else 0, 'total_s': wall}

PROMPTS = [
    "What is the capital of France?",
    "Write a short poem about the ocean.",
    "Explain how a transformer model works in simple terms.",
    "Write a Python function to compute fibonacci numbers.",
    "If a train travels at 60mph for 2.5 hours, how far does it go?",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='Path to .gguf model file')
    ap.add_argument('--ollama-model', default='gemma3:1b')
    ap.add_argument('--max-tokens', type=int, default=64)
    ap.add_argument('--prompt')
    ap.add_argument('--suite', action='store_true')
    ap.add_argument('--skip-ollama', action='store_true')
    ap.add_argument('--skip-adam', action='store_true')
    ap.add_argument('--output')
    args = ap.parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║        ADAM vs Ollama — GGUF Benchmark                  ║")
    print("╚══════════════════════════════════════════════════════════╝")

    ar = None
    if not args.skip_adam:
        print("[INIT] Loading ADAM...")
        try: ar = ADAMRunner(args.model); ar.load()
        except Exception as e:
            print(f"[INIT] ADAM failed: {e}")
            import traceback; traceback.print_exc(); ar = None

    olr = None
    if not args.skip_ollama:
        olr = OllamaRunner(args.ollama_model)
        if olr.check(): print(f"[INIT] Ollama ready ({args.ollama_model})")
        else: print(f"[INIT] Ollama '{args.ollama_model}' not found"); olr = None

    prompts = [args.prompt] if args.prompt else (PROMPTS if args.suite else PROMPTS[:1])
    results = []
    for i, prompt in enumerate(prompts):
        print(f"\n{'='*60}\nPROMPT {i+1}/{len(prompts)}: {prompt[:50]}...\n{'='*60}")
        row = {'prompt': prompt}
        for label, runner in [('adam', ar), ('ollama', olr)]:
            if runner is None: continue
            print(f"  [{label}] Running...")
            try:
                r = runner.run(prompt, args.max_tokens)
                row[label] = r
                tps = r.get('decode_tps', 0)
                print(f"  [{label}] done {tps:.1f} tok/s | {r.get('text','')[:60]}")
            except Exception as e:
                print(f"  [{label}] error: {e}")
                row[label] = {'error': str(e)}
        results.append(row)

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"{'#':>3} | {'ADAM GPU':>12} | {'Ollama':>12} | {'Speedup':>8}")
    print(f"{'-'*3}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}")
    for i, r in enumerate(results):
        ag = r.get('adam', {}).get('decode_tps', 0)
        ol = r.get('ollama', {}).get('decode_tps', 0)
        sp = f"{ag/ol:.2f}x" if ol > 0 and ag > 0 else "N/A"
        print(f"{i+1:>3} | {ag:>10.1f}/s | {ol:>10.1f}/s | {sp:>8}")

    if args.output:
        with open(args.output, 'w') as f: json.dump(results, f, indent=2, default=str)
        print(f"\nSaved: {args.output}")

if __name__ == '__main__': main()
