"""ADAM — GGUF Tokenizer (SentencePiece-compatible, works with any GGUF vocab)."""
import re
import numpy as np
from typing import List, Optional

class GGUFTokenizer:
    def __init__(self, vocab=None, scores=None, bos_id=2, eos_id=1,
                 sp_model_path=None, token_types=None, add_space_prefix=True):
        self.bos_id, self.eos_id = bos_id, eos_id
        self._sp = None; self._vocab = vocab; self._t2i = {}
        self._specials = {}   # special token string → ID (GGUF token_type == 3)
        self._special_re = None
        # When False (e.g. Gemma3: tokenizer.ggml.add_space_prefix=False), do NOT
        # prepend ▁ to plain-text segments — only replace internal spaces with ▁.
        self._add_space_prefix = add_space_prefix
        if sp_model_path:
            try:
                import sentencepiece as spm
                self._sp = spm.SentencePieceProcessor(); self._sp.Load(sp_model_path)
                return
            except: pass
        if vocab:
            self._t2i = {t: i for i, t in enumerate(vocab)}
            # Build a fast-lookup map for CONTROL tokens (type=3).
            # These must be emitted as single token IDs — never fed through BPE,
            # because the ▁-prefix normalization would corrupt them.
            if token_types:
                for i, (tok, ttype) in enumerate(zip(vocab, token_types)):
                    if ttype == 3 and tok:
                        self._specials[tok] = i
            if self._specials:
                # Compile once: longest match first to avoid prefix conflicts.
                pattern = '(' + '|'.join(
                    re.escape(s) for s in sorted(
                        self._specials.keys(), key=len, reverse=True)
                ) + ')'
                self._special_re = re.compile(pattern)

    def encode(self, text, add_bos=True):
        if self._sp:
            ids = self._sp.Encode(text)
            return ([self.bos_id] + ids) if add_bos else ids
        if self._vocab: return self._encode_bpe(text, add_bos)
        ids = list(text.encode('utf-8'))
        return ([self.bos_id] + ids) if add_bos else ids

    def decode(self, ids):
        if self._sp: return self._sp.Decode([i for i in ids if i not in (self.bos_id, self.eos_id)])
        if self._vocab:
            special_ids = set(self._specials.values()) if self._specials else set()
            parts = []
            for i in ids:
                if i in special_ids: continue  # skip all type-3 tokens (<bos>, <eos>, <end_of_turn>, …)
                if 0 <= i < len(self._vocab):
                    parts.append(self._vocab[i].replace('▁', ' '))
            return ''.join(parts)
        return bytes(i for i in ids if 0 <= i < 256).decode('utf-8', errors='replace')

    def _encode_bpe(self, text, add_bos):
        ids = [self.bos_id] if add_bos else []
        if self._special_re:
            # Split on known control tokens, preserving the delimiters.
            # re.split with a capturing group includes the matched tokens in the list.
            parts = self._special_re.split(text)
            for part in parts:
                if not part:
                    continue
                if part in self._specials:
                    ids.append(self._specials[part])
                else:
                    ids.extend(self._bpe_segment(part))
        else:
            ids.extend(self._bpe_segment(text))
        return ids

    def _bpe_segment(self, text):
        """Greedy BPE encode a plain text segment (contains no special tokens).

        Spaces are replaced with ▁.  A leading ▁ is prepended to signal a word
        boundary — but ONLY when the segment starts with a word character.
        Segments that start with whitespace (\n, \t …) or are already prefixed
        with ▁ do NOT get an extra leading ▁.  This prevents a spurious ▁ token
        before '\n' segments that sit between two special tokens (e.g.
        <end_of_turn>\\n<start_of_turn>) while keeping '▁user', '▁Hello' etc.
        working correctly for every model.
        """
        ids = []
        text = text.replace(' ', '▁')
        if not text:
            return ids
        # Prepend word-boundary marker only for word-like content.
        first = text[0]
        if self._add_space_prefix and first not in ('\n', '\r', '\t', '\f', '▁'):
            text = '▁' + text
        i = 0
        while i < len(text):
            best = None
            for l in range(min(32, len(text)-i), 0, -1):
                if text[i:i+l] in self._t2i:
                    best = (l, self._t2i[text[i:i+l]]); break
            if best: ids.append(best[1]); i += best[0]
            else:
                for b in text[i].encode('utf-8'):
                    bt = f'<0x{b:02X}>'
                    ids.append(self._t2i.get(bt, 0))
                i += 1
        return ids

    @property
    def vocab_size(self):
        if self._sp: return self._sp.GetPieceSize()
        return len(self._vocab) if self._vocab else 256
