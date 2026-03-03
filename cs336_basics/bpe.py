import regex as re
from typing import Dict, List, Tuple, BinaryIO
import time
import os

class ByteLevelBPE:
  def __init__(self):
    """Record and return vocab and merges in train."""
    self.vocab: Dict[int, bytes] = {}
    self.merges: List[Tuple[bytes, bytes]] = []

  @staticmethod
  def _get_gpt2_pattern() -> str:
    """GPT-2 style regex pattern."""
    return r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

  @staticmethod
  def _find_chunk_boundaries(
      file: BinaryIO,
      desired_num_chunks: int,
  ) -> list[int]:
      """
      Simple chunk boundaries - just split by byte size without caring about token boundaries.
      """
      # Get total file size in bytes
      file.seek(0, os.SEEK_END)
      file_size = file.tell()
      file.seek(0)

      if file_size == 0:
          return [0]

      chunk_size = file_size // desired_num_chunks
      
      # Create boundaries at regular intervals
      boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
      boundaries[-1] = file_size
      
      return boundaries

  def train(
    self,
    input_path: str,
    vocab_size: int,
    special_tokens: List[str]
  ) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """Train bpe tokenizer."""
    file_size = os.path.getsize(input_path)
    
    # 对于小文件，直接使用简单方法
    if file_size < 100 * 1024 * 1024:  # 小于100MB
      with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
        
        # freq, vocab, merge
        freq = self._pretokenize_and_count(text, special_tokens)
        self._init_vocab(special_tokens)
        self._merge(freq, vocab_size)
        
        return self.vocab, self.merges
    
    # 对于大文件，使用分块处理
    with open(input_path, "rb") as f:
        # 确定分块边界，每块大约50MB
        desired_num_chunks = max(1, file_size // (50 * 1024 * 1024))
        boundaries = self._find_chunk_boundaries(f, desired_num_chunks)
        
        # 合并所有块的词频统计
        combined_freq = {}
        
        # 逐块处理
        for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            f.seek(start)
            chunk_bytes = f.read(end - start)
            
            # 解码为文本
            try:
                chunk_text = chunk_bytes.decode("utf-8")
            except UnicodeDecodeError:
                # 如果解码失败，尝试忽略错误，但要小心边界问题
                # 找到最后一个完整字符的边界
                chunk_text = chunk_bytes.decode("utf-8", errors="ignore")
            
            # 对此块进行预分词并统计词频
            chunk_freq = self._pretokenize_and_count(chunk_text, special_tokens)
            
            # 合并词频
            for word, count in chunk_freq.items():
                combined_freq[word] = combined_freq.get(word, 0) + count
        
        # 使用合并后的词频进行BPE训练
        self._init_vocab(special_tokens)
        self._merge(combined_freq, vocab_size)
        
        return self.vocab, self.merges

  def _pretokenize_and_count(self, text, special_tokens):
    """Init and return freq of the words."""
    # eg. { (b'h',b'e',b'll',b'o'): 64, (b' ',b'n',b'i',b'h',b'a',b'o'): 233}
    
    # 先按特殊token分割
    if special_tokens:
        # 确保特殊token按长度降序排序，避免匹配问题
        sorted_special = sorted(special_tokens, key=len, reverse=True)
        special_pattern = "|".join([re.escape(token) for token in sorted_special])
        # 使用split保留特殊token作为分隔符，但不保留在结果中
        # 注意：re.split在有capture group时会保留分隔符，这里我们用non-capturing group
        blocks = re.split(f"(?:{special_pattern})", text)
    else:
        blocks = [text]
    
    pattern = self._get_gpt2_pattern()
    freq = {}

    for block in blocks:
        if not block:  # 跳过空块
            continue
        
        # 使用GPT-2模式分词
        for match in re.finditer(pattern, block):
            segment = match.group()
            if not segment:
                continue
            
            # 将每个字符转为字节
            bytes_data = tuple(bytes([b]) for b in segment.encode('utf-8'))
            freq[bytes_data] = freq.get(bytes_data, 0) + 1
    
    return freq

  def _init_vocab(self, special_tokens):
    """Init vocab."""
    self.vocab = {}
    self.merges = []

    # 添加特殊token
    for i, special_token in enumerate(special_tokens):
        self.vocab[i] = special_token.encode('utf-8')

    # 添加256个字节token
    base_index = len(special_tokens)
    for i in range(256):
        self.vocab[base_index + i] = bytes([i])

  def _merge(self, freq, target_vocab_size):
    """Perform bpe merges."""
    cur_vocab_size = len(self.vocab)
    
    # 限制合并次数，避免无限循环
    max_merges = min(target_vocab_size - cur_vocab_size, 50000)
    
    for merge_idx in range(max_merges):
        pair_freq = self._get_pair_freq(freq)
        if not pair_freq:
            break
            
        # 找到频率最高的pair
        best_pair_freq = max(pair_freq.items(), key=lambda x: (x[1], x[0]))
        best_pair = best_pair_freq[0]
        
        self.merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        self.vocab[len(self.vocab)] = new_token
        
        # 更新freq字典
        freq = self._apply_merge(freq, best_pair)

  def _get_pair_freq(self, freq):
      pair_freq = {}
      for tokens, count in freq.items():
          for i in range(len(tokens) - 1):
              pair = (tokens[i], tokens[i + 1])
              pair_freq[pair] = pair_freq.get(pair, 0) + count
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
        
        if new_tokens:  # 确保不为空
            new_freq[tuple(new_tokens)] = new_freq.get(tuple(new_tokens), 0) + count
    
    return new_freq