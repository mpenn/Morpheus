import collections

import cudf
import cupy as cp

Feature = collections.namedtuple(  # pylint: disable=invalid-name
    "Feature",
    ["input_ids", "input_mask", "segment_ids"]
)

### Model loading utils
def create_vocab_table(vocabpath):
    """
        Create Vocabulary tables from the vocab.txt file
        
        Parameters:
        ___________
        vocabpath: Path of vocablary file
        Returns:
        ___________
        id2vocab: np.array, dtype=<U5
        vocab2id: dict that maps strings to int
    """
    id2vocab = []
    vocab2id = {}
    import numpy as np
    with open(vocabpath) as f:
        for index, line in enumerate(f):
            token = line.split()[0]
            id2vocab.append(token)
            vocab2id[token] = index
    return np.array(id2vocab), vocab2id

def tokenize_text_series(text_ser, seq_len, stride, vocab_hash_file):
    """
        This function tokenizes a text series using the bert subword_tokenizer and vocab-hash
        
        Parameters
        __________

        text_ser: Text Series to tokenize
        seq_len: Sequence Length to use (We add to special tokens for ner classification job)
        stride : Stride for the tokenizer
        vocab_hash_file: vocab_hash_file to use (Created using `perfect_hash.py` with compact flag)

        Returns
        _______
         A dictionary with these keys {'token_ar':,'attention_ar':,'metadata':}

    """
    #print("Tokenizing Text Series")
    if len(text_ser) == 0:
        return Feature(None, None, None)

    max_rows_tensor = len(text_ser) * 2
    max_length = seq_len

    auto_stride = seq_len // 2
    auto_stride = auto_stride + auto_stride // 2

    tokens, attention_masks, metadata = text_ser.str.subword_tokenize(
        vocab_hash_file,
        do_lower=False,
        max_rows_tensor=max_rows_tensor,
        stride=stride,
        max_length=max_length,
        do_truncate=False,
    )
    del text_ser
    ### reshape metadata into a matrix
    metadata = metadata.reshape(-1, 3)
    tokens = tokens.reshape(-1, max_length)
    ## Attention mask
    attention_masks = attention_masks.reshape(-1, max_length)
    
    output_f=Feature(input_ids=tokens,
            input_mask =attention_masks,
            segment_ids = metadata)    
        
    return output_f
