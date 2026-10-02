"""
Fine-tuning BERT phishing (v6 FINAL) — Équilibre optimal
=========================================================
- Fix LayerNorm garanti (renommage beta/gamma -> bias/weight)
- Hyperparamètres conservateurs (basés sur v4, les meilleurs résultats)
- Dashboard complet (loss + F1 + precision/recall)
- Comparaison automatique des checkpoints sur test set
"""

import json
import gc
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)
import torch
import torch.nn as nn

import datasets.formatting.torch_formatter as _torch_formatter
_torch_formatter.config.TORCHVISION_AVAILABLE = False

# ------------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------------
DATASET_PATH = "merged_dataset_balanced.jsonl"
CHECKPOINT_PT = "phishing-bert-20230517.pt"
TOKENIZER_NAME = "bert-base-uncased"
OUTPUT_DIR = "./phishing-bert-finetuned-v6"
MAX_LENGTH = 128
RANDOM_SEED = 42

# Hyperparamètres CONSERVATEURS (basés sur v4, meilleurs résultats)
LR = 1e-5
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
EPOCHS = 5
WEIGHT_DECAY = 0.01
PATIENCE = 2

set_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# ------------------------------------------------------------------
# 2. Dataset
# ------------------------------------------------------------------
print("Chargement du dataset...")
records = []
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

df = pd.DataFrame(records)
print(f"Total: {len(df)} | phishing={df['label'].sum()} ({df['label'].mean()*100:.1f}%)")

train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=RANDOM_SEED)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=RANDOM_SEED)
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))
test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

# ------------------------------------------------------------------
# 3. Tokenisation
# ------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
test_ds = test_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

for ds in [train_ds, val_ds, test_ds]:
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# ------------------------------------------------------------------
# 4. Modèle — FIX LayerNorm + chargement robuste
# ------------------------------------------------------------------
print(f"\nChargement du checkpoint {CHECKPOINT_PT}...")

checkpoint_data = torch.load(CHECKPOINT_PT, map_location="cpu", weights_only=False)
state_dict = checkpoint_data.state_dict() if hasattr(checkpoint_data, 'state_dict') else checkpoint_data
del checkpoint_data
gc.collect()

# --- FIX LayerNorm : renommage beta/gamma -> bias/weight ---
renamed_state_dict = {}
ln_renamed = 0
for key, value in state_dict.items():
    new_key = key
    if "LayerNorm.beta" in key:
        new_key = key.replace("LayerNorm.beta", "LayerNorm.bias")
        ln_renamed += 1
    elif "LayerNorm.gamma" in key:
        new_key = key.replace("LayerNorm.gamma", "LayerNorm.weight")
        ln_renamed += 1
    renamed_state_dict[new_key] = value

print(f"  -> {ln_renamed} paramètres LayerNorm renommés")

# Construction du modèle (dropout par défaut BERT = 0.1, ne pas toucher)
model = AutoModelForSequenceClassification.from_pretrained(TOKENIZER_NAME, num_labels=2)

# Chargement des poids
missing, unexpected = model.load_state_dict(renamed_state_dict, strict=False)

# Filtrer les warnings attendus
expected_unexpected = {'cls.predictions.bias', 'cls.predictions.transform.dense.bias',
                       'cls.seq_relationship.weight', 'cls.predictions.transform.LayerNorm.weight',
                       'cls.predictions.transform.dense.weight', 'cls.predictions.transform.LayerNorm.bias',
                       'cls.seq_relationship.bias', 'bert.embeddings.position_ids'}
unexpected_real = [k for k in unexpected if k not in expected_unexpected]
missing_real = [k for k in missing if not k.startswith('cls.predictions')]

if missing_real:
    print(f"⚠️ MANQUANTS: {missing_real[:3]}... ({len(missing_real)} total)")
if unexpected_real:
    print(f"⚠️ INATTENDUS: {unexpected_real[:3]}... ({len(unexpected_real)} total)")

# Vérifications
assert "classifier.weight" not in missing, "ERREUR: classifier non chargé !"
assert torch.equal(renamed_state_dict["classifier.weight"], model.classifier.weight.data)
assert torch.equal(renamed_state_dict["classifier.bias"], model.classifier.bias.data)

# Vérification LayerNorm
ln_key = "bert.embeddings.LayerNorm.weight"
if ln_key in renamed_state_dict:
    assert torch.equal(renamed_state_dict[ln_key], model.bert.embeddings.LayerNorm.weight.data)
    print("✅ LayerNorm: transfert vérifié")
print("✅ Checkpoint chargé avec succès.\n")

# ------------------------------------------------------------------
# 5. Classes déséquilibrées
# ------------------------------------------------------------------
class_weights = compute_class_weight('balanced', classes=np.unique(train_df['label']), y=train_df['label'])
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
print(f"Poids de classe: {class_weights}")

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        if self.class_weights is not None and torch.cuda.is_available():
            self.class_weights = self.class_weights.cuda()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

# ------------------------------------------------------------------
# 6. Métriques
# ------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1, zero_division=0
    )
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# ------------------------------------------------------------------
# 7. Entraînement
# ------------------------------------------------------------------
total_steps = (len(train_ds) // BATCH_SIZE) * EPOCHS
warmup_steps = int(0.1 * total_steps)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=WEIGHT_DECAY,
    warmup_steps=warmup_steps,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    logging_steps=50,
    logging_first_step=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=3,
    report_to="none",
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=0,
    seed=RANDOM_SEED,
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    class_weights=class_weights_tensor,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
)

print(f"Config: LR={LR}, WeightDecay={WEIGHT_DECAY}")
print(f"Early stopping: eval_loss | patience={PATIENCE}")
print(f"Warmup: {warmup_steps}/{total_steps} steps\n")
print("🚀 Début du fine-tuning...\n")
trainer.train()

# ------------------------------------------------------------------
# 8. Dashboard
# ------------------------------------------------------------------
def plot_dashboard(trainer, save_path):
    history = trainer.state.log_history

    train_logs = [x for x in history if 'loss' in x and 'eval_loss' not in x]
    val_logs = [x for x in history if 'eval_loss' in x]

    train_steps = list(range(len(train_logs)))
    train_losses = [x['loss'] for x in train_logs]

    val_epochs = [x['epoch'] for x in val_logs]
    val_losses = [x['eval_loss'] for x in val_logs]
    val_f1 = [x.get('eval_f1', 0) for x in val_logs]
    val_precision = [x.get('eval_precision', 0) for x in val_logs]
    val_recall = [x.get('eval_recall', 0) for x in val_logs]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Train loss
    ax = axes[0, 0]
    ax.plot(train_steps, train_losses, color='steelblue', alpha=0.3, linewidth=0.8)
    window = max(1, len(train_losses) // 50)
    if window > 1 and len(train_losses) > window:
        smoothed = np.convolve(train_losses, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(train_losses)), smoothed, color='steelblue', linewidth=2, label=f'Smoothed ({window})')
    ax.set_xlabel('Step'); ax.set_ylabel('Loss'); ax.set_title('Train Loss (par step)')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Train vs Val loss
    ax = axes[0, 1]
    steps_per_epoch = len(train_losses) // EPOCHS
    train_per_epoch = [np.mean(train_losses[i*steps_per_epoch:(i+1)*steps_per_epoch]) for i in range(EPOCHS)]
    ax.plot(range(1, EPOCHS+1), train_per_epoch, color='steelblue', marker='o', linewidth=2, label='Train')
    ax.plot(val_epochs, val_losses, color='crimson', marker='s', markersize=8, linewidth=2.5, label='Validation')
    if val_losses:
        min_idx = np.argmin(val_losses)
        ax.annotate(f'Min: {val_losses[min_idx]:.4f}\nEpoch {val_epochs[min_idx]}',
                   xy=(val_epochs[min_idx], val_losses[min_idx]),
                   xytext=(val_epochs[min_idx]+0.2, val_losses[min_idx]+0.008),
                   arrowprops=dict(arrowstyle='->', color='crimson'),
                   fontsize=10, color='crimson')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title('Train vs Validation Loss')
    ax.legend(); ax.grid(True, alpha=0.3)

    # F1
    ax = axes[1, 0]
    ax.plot(val_epochs, val_f1, color='green', marker='D', markersize=7, linewidth=2, label='F1')
    ax.axhline(y=max(val_f1), color='green', linestyle='--', alpha=0.5, label=f'Max F1: {max(val_f1):.4f}')
    ax.set_xlabel('Epoch'); ax.set_ylabel('F1 Score'); ax.set_title('F1 sur Validation')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Precision / Recall
    ax = axes[1, 1]
    ax.plot(val_epochs, val_precision, color='blue', marker='o', linewidth=2, label='Precision')
    ax.plot(val_epochs, val_recall, color='orange', marker='s', linewidth=2, label='Recall')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Score'); ax.set_title('Precision vs Recall (Validation)')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n📊 Dashboard: {save_path}")

plot_dashboard(trainer, os.path.join(OUTPUT_DIR, "training_dashboard.png"))

# ------------------------------------------------------------------
# 9. Comparaison checkpoints sur TEST set
# ------------------------------------------------------------------
print("\n" + "="*65)
print("COMPARAISON DES CHECKPOINTS SUR LE TEST SET")
print("="*65)

from transformers import AutoModelForSequenceClassification

checkpoints = [os.path.join(OUTPUT_DIR, d) for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
checkpoints.sort(key=lambda x: int(x.split("-")[-1]))

results = []
for ckpt in checkpoints:
    print(f"\nÉvaluation {os.path.basename(ckpt)}...")
    ckpt_model = AutoModelForSequenceClassification.from_pretrained(ckpt)
    ckpt_args = TrainingArguments(output_dir="/tmp/eval", report_to="none", per_device_eval_batch_size=EVAL_BATCH_SIZE)
    ckpt_trainer = Trainer(model=ckpt_model, args=ckpt_args, compute_metrics=compute_metrics)
    res = ckpt_trainer.evaluate(test_ds)
    results.append((os.path.basename(ckpt), res['eval_loss'], res.get('eval_f1', 0),
                   res.get('eval_accuracy', 0), res.get('eval_precision', 0), res.get('eval_recall', 0)))

print("\n" + "-"*65)
print(f"{'Checkpoint':<18} {'Loss':>8} {'F1':>8} {'Acc':>8} {'Prec':>8} {'Rec':>8}")
print("-"*65)
best_loss_idx = min(range(len(results)), key=lambda i: results[i][1])
for i, (name, loss, f1, acc, prec, rec) in enumerate(results):
    marker = " <-- BEST" if i == best_loss_idx else ""
    print(f"{name:<18} {loss:>8.4f} {f1:>8.4f} {acc:>8.4f} {prec:>8.4f} {rec:>8.4f}{marker}")

# ------------------------------------------------------------------
# 10. Évaluation finale (meilleur modèle)
# ------------------------------------------------------------------
print(f"\n🏆 Meilleur checkpoint (val_loss): {trainer.state.best_model_checkpoint}")
print(f"   Score validation: {trainer.state.best_metric:.6f}")

print("\n📋 Évaluation finale sur test set...\n")
test_results = trainer.evaluate(test_ds)
print(test_results)

preds_output = trainer.predict(test_ds)
preds = np.argmax(preds_output.predictions, axis=-1)
labels = preds_output.label_ids
cm = confusion_matrix(labels, preds)

print("\nMatrice de confusion:")
print("                 Prédit: Légitime | Prédit: Phishing")
print(f"Réel: Légitime      {cm[0][0]:>6}       |      {cm[0][1]:>6}")
print(f"Réel: Phishing       {cm[1][0]:>6}       |      {cm[1][1]:>6}")
print(f"\n🔴 Faux négatifs = {cm[1][0]} | 🟡 Faux positifs = {cm[0][1]}")

# ------------------------------------------------------------------
# 11. Sauvegarde
# ------------------------------------------------------------------
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n💾 Modèle final sauvegardé dans: {OUTPUT_DIR}")