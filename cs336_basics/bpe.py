import regex as re
from typing import Dict, List, Tuple
import time

class ByteLevelBPE:
  def __init__(self):
    """Record and return vocab and merges in train."""
    self.vocab: Dict[int, bytes] = {}
    self.merges: List[Tuple[bytes, bytes]] = []

  @staticmethod
  def _get_gpt2_pattern() -> str:
    """GPT-2 style regex pattern."""
    return r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

  def train(
    self,
    input_path: str,
    vocab_size: int,
    special_tokens: List[str]
  ) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """Train bpe tokenizer."""
    with open(input_path, "r", encoding="utf-8") as f:
      text = f.read()

      # freq, vocab, merge
      freq = self._pretokenize_and_count(text, special_tokens)
      self._init_vocab(special_tokens)
      self._merge(freq, vocab_size)

      return self.vocab, self.merges

  def _pretokenize_and_count(self, text, special_tokens):
    """Init and return freq of the words."""
    # eg. {' hello': 64, ' nihao': 233}

    # First split by special pattern,
    # then split by gpt2 pattern.
    pattern = self._get_gpt2_pattern()
    special_pattern = "|".join([re.escape(token) for token in special_tokens])
    blocks = re.split(special_pattern, text)
    freq = {}

    for block in blocks:
      for match in re.finditer(pattern, block):
        segment = match.group()
        bytes_data = [bytes([b]) for b in segment.encode('utf-8')]
        #print(f'segment: {segment}, bytes_data: {bytes_data}')
        if tuple(bytes_data) not in freq:
          freq[tuple(bytes_data)] = 1
        else:
          freq[tuple(bytes_data)] += 1
    return freq

  def _init_vocab(self, special_tokens):
    """Init vocab."""
    self.vocab = {}
    self.merges = []

    for i, special_token in enumerate(special_tokens):
      #bytes_token = [bytes([b]) for b in special_token.encode('utf-8')]
      self.vocab[i] = special_token.encode('utf-8')

    base_index: int = len(special_tokens)
    for i in range(256):
      self.vocab[base_index + i] = bytes([i])
    #print('vocab: ', self.vocab)

  def _merge(self, freq, target_vocab_size):
    """Perform bpe merges."""
    cur_vocab_size = len(self.vocab)

    for _ in range(cur_vocab_size, target_vocab_size):
      pair_freq = self._get_pair_freq(freq)
      if not pair_freq:
        break
      best_pair_freq = max(pair_freq.items(), key=lambda x: (x[1], x[0]))
      best_pair = best_pair_freq[0]
      self.merges.append(best_pair)
      new_token = best_pair[0] + best_pair[1]
      self.vocab[len(self.vocab)] = new_token
      # Update freq dict
      freq = self._apply_merge(freq, best_pair)

      #print(f'merge: {best_pair}')

  def _get_pair_freq(self, freq):
      pair_freq = {}
      # accumulation.
      for tokens, count in freq.items():
        for i in range(len(tokens) - 1):
          pair = (tokens[i], tokens[i + 1])
          
          if pair in pair_freq:
            pair_freq[pair] += count
          else:
            pair_freq[pair] = count

      return pair_freq

  def _apply_merge(self, freq, pair):
    """Update pair in freq."""

    new_freq = {}
    for tokens, count in freq.items():
      new_tokens = []
      i = 0
      while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
          merged_token = pair[0] + pair[1]
          new_tokens.append(merged_token)
          i += 2
        else:
          new_tokens.append(tokens[i])
          i += 1
      
      new_freq[tuple(new_tokens)] = count

    return new_freq
