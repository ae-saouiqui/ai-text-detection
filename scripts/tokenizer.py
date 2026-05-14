
class Tokenizer:
    def __init__(self, model: str, use_fast: bool = True,max_length:int=None):
        # To rememeber :
        #  - Loading tokenizer is heavy  (despite his effiency because it written in Rust)
        #  - Try to import it lazily
        from transformers import AutoTokenizer
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model,use_fast=use_fast)


    def tokenize(self, examples):
        if self.max_length:
            tokens = self.tokenizer(examples["text"],truncation=True,padding="max_length",max_length=self.max_length)
        else:
            tokens = self.tokenizer(examples["text"],truncation=True,padding=False)
        tokens["labels"] = examples["label"]
        return tokens
    
    def save_pretrained(self, path: str):
        self.tokenizer.save_pretrained(path)
