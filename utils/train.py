import argparse

parser = argparse.ArgumentParser(description="Train a model on a dataset")
parser.add_argument(
    "--dataset",
    type=str,
    default="surrey-nlp/PLOD-CW",
    help="dataset to train on",
)

parser.add_argument(
    "--model_checkpoint",
    type=str,
    default="antoineedy/stanford-deidentifier-base-finetuned-ner",
    help="model checkpoint to train",
)

parser.add_argument(
    "--push_to_hub",
    type=bool,
    default=True,
    help="push the model to the hub",
)

parser.add_argument(
    "--num_train_epochs",
    type=int,
    default=60,
    help="number of training epochs",
)

parser.add_argument(
    "--learning_rate",
    type=float,
    default=2e-5,
    help="learning rate",
)

parser.add_argument(
    "--model_name",
    type=str,
    default="model",
    help="model name",
)

# take the arguments from the command line
args = parser.parse_args()

model_checkpoint = args.model_checkpoint
dataset_str = args.dataset

import transformers

batch_size = 16

from datasets import load_dataset, load_metric

datasets = load_dataset(dataset_str)

TEXT2ID = {
    "B-O": 0,
    "B-AC": 1,
    "B-LF": 2,
    "I-LF": 3,
}

datasets = datasets.map(lambda x: {"ner_tags": [TEXT2ID[tag] for tag in x["ner_tags"]]})


label_list = list(set(datasets["train"]["ner_tags"][0]))
label_list


from datasets import ClassLabel, Sequence
import random
import pandas as pd
from IPython.display import display, HTML


def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(
        dataset
    ), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
        elif isinstance(typ, Sequence) and isinstance(typ.feature, ClassLabel):
            df[column] = df[column].transform(
                lambda x: [typ.feature.names[i] for i in x]
            )
    display(HTML(df.to_html()))


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)


import transformers

assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)


label_all_tokens = True


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)


from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint, num_labels=len(label_list), ignore_mismatched_sizes=True
)


model_name = model_checkpoint.split("/")[-1]

args = TrainingArguments(
    f"{model_name}-finetuned-ner",
    evaluation_strategy="epoch",
    learning_rate=args.learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=args.num_train_epochs,
    weight_decay=0.01,
    push_to_hub=True,
)


from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer)


metric = load_metric("seqeval")


example = datasets["train"][0]
tokenized_input = tokenizer(example["tokens"], is_split_into_words=True)
tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
tokens


labels = [label_list[i] for i in example[f"ner_tags"]]
metric.compute(predictions=[labels], references=[labels])


import numpy as np


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


from sklearn.metrics import classification_report


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    tp = []
    for l in true_predictions:
        tp += l
    tl = []
    for l in true_labels:
        tl += l

    cr = classification_report(tl, tp, output_dict=True)
    out = {}
    for key in cr.keys():
        if key == "accuracy":
            out[key] = cr[key]
        else:
            for new_k in ["precision", "recall", "f1-score"]:
                out[key + "_" + new_k] = cr[key][new_k]
    return out


import torch
from torch import nn


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        w3 = [0.0443, 0.6259, 1.0000, 0.4525]
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(w3).to(logits.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


trainer = CustomTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


trainer.train()


trainer.evaluate()


predictions, labels, _ = trainer.predict(tokenized_datasets["validation"])
predictions = np.argmax(predictions, axis=2)

# Remove ignored index (special tokens)
true_predictions = [
    [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]
true_labels = [
    [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

results = metric.compute(predictions=true_predictions, references=true_labels)
results


print(true_labels)

labels = []
for l in true_labels:
    labels += l
pred = []
for l in true_predictions:
    pred += l

ID2TEXT = {0: "B-O", 1: "B-AC", 2: "B-LF", 3: "I-LF"}
TEXT2ID = {v: k for k, v in ID2TEXT.items()}
labels = [ID2TEXT[k] for k in labels]
pred = [ID2TEXT[k] for k in pred]
print(labels)


# confusion matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

alabels = ["B-O", "B-AC", "B-LF"]
nlabels = ["O", "Abb.", "Long-forms"]

c = labels
p = pred


trainer.push_to_hub(args.model_name)
