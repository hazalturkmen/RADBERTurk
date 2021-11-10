from tokenizers import BertWordPieceTokenizer

tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=False,
    strip_accents=False,
    lowercase=False,
)

print("Training is started...")


trainer = tokenizer.train(
    r"C:\Users\AI User\Desktop\TurkRADBERT\corpus\all.txt",
    vocab_size=32000,
    min_frequency=2,
    show_progress=True,
    special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
    limit_alphabet=1000,
    wordpieces_prefix="##"
)
print("Training is over...")

tokenizer.save("./", "cased")