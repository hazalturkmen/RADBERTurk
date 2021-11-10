from tokenizers import BertWordPieceTokenizer

tokenizer_cased = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=False,
    strip_accents=False,
    lowercase=False,
)

trainer_cased = tokenizer_cased.train(
    "/home/hazal/nlp_dataset/corpus/all.txt",
    vocab_size=32000,
    min_frequency=2,
    show_progress=True,
    special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
    limit_alphabet=1000,
    wordpieces_prefix="##"
)

tokenizer_cased.save("./", "cased")


tokenizer_uncased = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=False,
    strip_accents=False,
    lowercase=True,
)

trainer_uncased = tokenizer_uncased.train(
    "tr_final",
    vocab_size=32000,
    min_frequency=2,
    show_progress=True,
    special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
    limit_alphabet=1000,
    wordpieces_prefix="##"
)

tokenizer_uncased.save("./", "uncased")
