from transformers import BertTokenizer,BertModel

tokenizer = BertTokenizer.from_pretrained("pretrain_models/sikuroberta_vocabtxt")
model = BertModel.from_pretrained("pretrain_models/sikuroberta_vocabtxt")
print(model)

  