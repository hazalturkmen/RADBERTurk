import torch
import datetime
import time
from transformers import BertConfig
from transformers import BertModel, BertTokenizer
from transformers import BertForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments


def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))


print("Is cuda available?")
print(torch.cuda.is_available())
print("----------------------------")

config = BertConfig(
    vocab_size=32_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=12,
    type_vocab_size=1,
    attention_probs_dropout_prob=0.1,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    hidden_size=768
)

tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')
model = BertForMaskedLM(config=config)
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="/home/hazal/nlp_dataset/brain_CT/tr_medical_processed.txt",
    block_size=512,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="./BioBERTRcased",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_gpu_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

torch.cuda.empty_cache()
print(torch.cuda.memory_summary(device=None, abbreviated=False))
print("Training is started...")
total = time.time()
trainer.train()
total_train_training_time = format_time(time.time() - total)
print("/n Total training time = " + total_train_training_time)
print("Model is saved...")
trainer.save_model("./trmedicalBERT")
