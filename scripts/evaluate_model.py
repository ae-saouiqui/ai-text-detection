from argparse import ArgumentParser
import json
import sys
from pathlib import Path
import os 
from dotenv import load_dotenv
import evaluate
from datasets import load_dataset
from transformers import (
AutoModelForSequenceClassification,
DataCollatorWithPadding
)
from tokenizer import Tokenizer
import numpy as np
import torch
from sklearn.metrics import (
    roc_curve,
    auc,
    classification_report
)

from utils import plot_roc_curve,plot_confusion_matrix
import json
from tqdm import tqdm
from torch.utils.data import DataLoader

load_dotenv()
test_path  = Path(os.getenv("TEST_PATH"))
FORMAT = os.getenv("FORMAT")
result_path = Path("../results")





def evaluate_per_lang(mask,metrics):
    res = metrics.compute(
        references=y_true[mask],
        predictions=y_pred[mask]
    )

    fpr, tpr, _ = roc_curve(y_true[mask], y_probs[mask])
    roc_auc = auc(fpr, tpr)

    res.update({"roc_auc": float(roc_auc)})

    return res




parser = ArgumentParser()

parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="Choose the model to fine-tune (The name must be exist in the model_map.json)"
)

parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Optional checkpoint path (e.g. checkpoint-2000)"
)


args = parser.parse_args()



try :
    with open("../configs/model_map.json","r") as file :
        models = json.load(file)
    # load the threshold 
    with open(result_path / "threhsolds.json","r") as f:
        thresholds = json.load(f)

    base = models[args.model]["trained_model"]
    threshold = thresholds[args.model]
    model_name = base + "/" + args.checkpoint if args.checkpoint else base
    # load the dataset 


    print("Load the test data")
    raw_data = load_dataset(
        FORMAT,
        data_files={"test":str(test_path)},
        columns=["text","label","lang"]
    ).shuffle(seed=42)
    langs= raw_data["test"]["lang"]


    print("Load the test metrics")
    # Load metrics
    metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

    print("Load the tokenizer and the model ")
    # Load tokenizer and model 
    tokenizer = Tokenizer(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    print("Start tokenizing")
    # To rememeber the DataLoader doesn't ignore colunms like Trainer class 
    # It only accepts 
    data = raw_data["test"].map(
        tokenizer.tokenize,
        remove_columns=raw_data["test"].column_names
        )

    data_collator = DataCollatorWithPadding(tokenizer.tokenizer)
    data.set_format("torch")
    data_loader = DataLoader(
        data,
        batch_size=16,
        collate_fn=data_collator,
        shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    y_true = []
    y_pred = []
    y_probs = []
    
    model.to(device)
    model.eval()
    print("Start testing ")


    for batch in tqdm(data_loader,desc="Tes",unit="batch"):
            
        inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device)
        }
        labels = batch["labels"]

        with torch.inference_mode():
            output = model(**inputs)
            
        logits = output.logits

        probs = torch.softmax(logits, dim=1)

        ai_probs = probs[:, 1]
        preds = (ai_probs > threshold).int()
        y_probs.extend(ai_probs.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)
    langs = np.array(langs)

    print("Compute metrics")
    results = metrics.compute(
        references = y_true,
        predictions = y_pred
    )

    fpr,tpr,_ = roc_curve(y_true,y_probs)
    roc_auc = auc(fpr, tpr)
    results.update({"roc_auc":float(roc_auc)})  


    # Evaluate based on rach language 
    en_mask = langs == "en"
    fr_mask = langs == "fr"

    # Setting the langugaes in a dictionnary
    all_results = {
        "global":results,
        "english":evaluate_per_lang(en_mask,metrics),
        "french":evaluate_per_lang(fr_mask,metrics)
    }


    result_folder = result_path / args.model / args.checkpoint if args.checkpoint else result_path / args.model
    result_folder.mkdir(parents=True, exist_ok=True)
    print("Plotting results")
    plot_roc_curve(fpr,tpr,roc_auc,result_folder=result_folder)
    plot_confusion_matrix(y_true,y_pred,result_folder=result_folder)
    print("Saving the results ")
    with open(result_folder / "metrics.json", "w") as f:
        json.dump(all_results, f, indent=4)

    print(classification_report(
        y_true,
        y_pred
    ))
    print("Test finished")
except KeyError as ke:
    print("The model you are trying to load doesn't exist ")
    print(f"Here is available models {list(models.keys())}")
    sys.exit(1)


