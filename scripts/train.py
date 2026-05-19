from argparse import ArgumentParser
from tokenizer import Tokenizer
import json
import sys
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
    )

from peft import(
    LoraConfig,
    TaskType,
    get_peft_model
)
from utils import compute_metrics
import numpy as np
import torch
import random
from dotenv import load_dotenv
import os 
from huggingface_hub import login


def main():
    load_dotenv()
    hf_token =  os.getenv("HF_TOKEN")
    train_path = Path(os.getenv("TRAINING_PATH"))
    validation_path = Path(os.getenv("VALIDATION_PATH"))
    FORMAT = os.getenv("FORMAT")
    
    parser = ArgumentParser()
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Choose the model to fine-tune (The name must be exist in the model_map.json)"
        )
    parser.add_argument(
        "--max_length",
        type=int,
        required=False,
        default=None,
        help="The max length must be an integer"
        )
    
    parser.add_argument(
        "-lr",
        "--lora",
        # Activate lora if it is flagged
        action="store_true",
        help="Enable LoRA fune-tuning (ensure you set its configuration in the lora.json.json file)"
    )
    args = parser.parse_args()

    try:
        with open("../configs/model_map.json","r") as file :
            models = json.load(file)
        with open("../configs/hp.json","r") as file :
            hps = json.load(file)
            
        model_name = models[args.model.strip()]["hf_model"]
        hp = hps[args.model.strip()]
        # Login to HF
        login(hf_token)
        seed = hp.get("seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


        # Load the data 

        # Just to remember :
        #     - Parquet supports predicate pushdown 
        #     - In the official HF documentation the load_dataset() doesn't have a "columns" property
        #     - columns = is passed via **kwargs (named kwargs_config or something related ) to Arrow/Parquet backend.
        #     - Arrow/Parquet Layer is responsible to select only the needed columns 

        print(" Loadin the dataset ".center(50,"*"))
        data = load_dataset(
            FORMAT,
            data_files={
                "train":str(train_path),
                "validation":str(validation_path)
                },
                columns=["text","label"]
                )

        # load the tokenizer 
        print(" Loadin the toknizer ".center(50,"*"))
        max_length = args.max_length
        tokenizer = Tokenizer(model_name,True,max_length)
        data = data.map(tokenizer.tokenize,batched = True)
        data = data.shuffle(seed=seed)
        # load the model
        print(" Loadin the model ".center(50,"*"))
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels = 2
            )
        # Applying LoRA 
        if args.lora:
            with open("../configs/lora.json","r") as f :
                loras = json.load(f)
            lora = loras[args.model.strip()]
            lora_config= LoraConfig(
                task_type=TaskType.SEQ_CLS,
                **lora
            ) 
            print("Applying LoRA fine-tuning".center(50,"*"))
            model = get_peft_model(model,lora_config)
        
        # Colating data (ensure dynamic padding for efficient memory usage)
        data_collator = DataCollatorWithPadding(tokenizer.tokenizer)
        # Setting the hyperparameters of traning
        print(" Setting ARguments ".center(50,"*"))
        training_arg = TrainingArguments(**hp)
        # set the trainer 
        trainer = Trainer(
            model = model,
            args = training_arg,
            data_collator=data_collator,
            train_dataset=data["train"],
            eval_dataset=data["validation"],
            compute_metrics=compute_metrics
            )
        print(" Start training ".center(50,"*"))
        trainer.train()
        print("Saving the model")
        trainer.save_model()
        tokenizer.save_pretrained(hp["output_dir"])
        print(" Training Done ".center(50,"*"))
    except KeyError as ke:
        print("The model you are trying to load doesn't exist ")
        print(f"Here is available models {list(models.keys())}")
        sys.exit(1)

if __name__  == "__main__":
    import torch.multiprocessing as mp 
    mp.set_start_method("spawn", force=True)
    main()

