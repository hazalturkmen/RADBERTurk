import pandas as pd
import torch
from torch.utils.data import Dataset


class Radataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len):
        self.df = pd.read_excel(csv_path)
        self.indices = self.df.index.tolist()
        self.df['sonuc'] = self.df[['Intraventriküler', 'Gliozis', 'Epidural',

       'Hidrosefali', 'Ensefalomalazi-Ansefalomalazi-Ansafalomalazi', 'Kronik iskemik değişiklikler','Lakün','Lökoriazis',
       'Mega sisterna magna','Menenjiom','SAK','Subdural','YOK']].values.tolist()
        self.len = len(self.df)
        self.data = self.df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        rapor = str(self.data.rapor[index])
        rapor = " ".join(rapor.split())
        inputs = self.tokenizer.encode_plus(
            rapor,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'labels': torch.tensor(self.data.sonuc[index], dtype=torch.long)
        }

    def __len__(self):
        return self.len
    