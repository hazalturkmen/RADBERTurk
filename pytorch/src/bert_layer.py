import torch
from transformers import BertModel


class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = BertModel.from_pretrained("dbmdz/bert-base-turkish-cased")
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 3)

    def forward(self, input_ids, attention_mask):
        _, output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output





