import torch
from transformers import BertModel


class BertClass(torch.nn.Module):

    def __init__(self, freeze_bert=True):
        super(BertClass, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 50, 3

        # Instantiate TR BERT model
        self.bert = BertModel.from_pretrained('dbmdz/bert-base-turkish-cased')

        # Instantiate an one-layer feed-forward classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            torch.nn.ReLU(),
            # nn.Dropout(0.5),
            torch.nn.Linear(H, D_out)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits
