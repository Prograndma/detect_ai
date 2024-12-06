import torch
import transformers
import os


class BERTClassifier(torch.nn.Module):
    def __init__(self, base_filename):
        self.base_filename = base_filename
        super(BERTClassifier, self).__init__()
        self.l1 = transformers.DistilBertModel.from_pretrained("distilbert/distilbert-base-uncased",
                                                     torch_dtype=torch.float32, attn_implementation="sdpa")
        # self.l1 = transformers.BertModel.from_pretrained("distilbert/distilbert-base-uncased")
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 1)

    def forward(self, inputs):
        output_1 = self.l1(**inputs, return_dict=False)[0][:, 0, :]
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

    def saved_model_exists(self, checkpoint):
        if not os.path.exists(f"{self.base_filename}/{checkpoint}"):
            return False
        return True

    def update_model_from_checkpoint(self, checkpoint):
        if not self.saved_model_exists(checkpoint):
            return "no saved model exists"

        path = f"{self.base_filename}/{checkpoint}"

        loaded = torch.load(path)

        return self.load_state_dict(loaded)

    def save(self, epoch):
        torch.save(self.state_dict(), f"{self.base_filename}/{epoch}")
