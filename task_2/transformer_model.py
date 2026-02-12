"""
ULTRA SIMPLE Transformer for Cybersecurity
NO TensorFlow required - uses PyTorch
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Try to import PyTorch, if not available use scikit-learn alternative
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    USE_TORCH = True
    print("✓ PyTorch loaded successfully")
except:
    USE_TORCH = False
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    print("⚠ PyTorch not found, using scikit-learn fallback")

# ============================================
# 1. GENERATE DATA
# ============================================

print("\n" + "=" * 60)
print("TRANSFORMER NETWORK FOR CYBERSECURITY")
print("=" * 60)

# Generate synthetic network traffic data
np.random.seed(42)
n_samples = 500

# Features: packet_size, duration, bytes_sent, bytes_received, protocol, flag
n_features = 6

# Normal traffic (70%)
X_normal = np.random.randn(int(n_samples * 0.7), n_features) * 0.5 + np.array([400, 1, 800, 1500, 0, 0])
# Attack traffic (30%)
X_attack = np.random.randn(int(n_samples * 0.3), n_features) * 1.5 + np.array([1000, 0.5, 3000, 500, 1, 2])

X = np.vstack([X_normal, X_attack])
y = np.hstack([np.zeros(len(X_normal)), np.ones(len(X_attack))])

# Shuffle
idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]

print(f"\n📊 Dataset: {len(X)} samples")
print(f"   • Normal: {sum(y == 0)} samples")
print(f"   • Attack: {sum(y == 1)} samples")
print(f"   • Features: {n_features}")

# ============================================
# 2. PREPROCESS
# ============================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"\n📂 Data split:")
print(f"   • Train: {len(X_train)} samples")
print(f"   • Test: {len(X_test)} samples")

# ============================================
# 3. TRANSFORMER MODEL (if PyTorch available)
# ============================================

if USE_TORCH:
    class SimpleTransformer(nn.Module):
        """Minimal transformer for classification"""

        def __init__(self, input_dim, hidden_dim=32):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads=2, batch_first=True)
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            self.norm2 = nn.LayerNorm(hidden_dim)
            self.classifier = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            # Add sequence dimension (seq_len=1)
            x = x.unsqueeze(1)
            x = self.input_proj(x)

            # Self-attention
            attn_out, _ = self.attention(x, x, x)
            x = self.norm1(x + attn_out)

            # Feed-forward
            ffn_out = self.ffn(x)
            x = self.norm2(x + ffn_out)

            # Pool and classify
            x = x.mean(dim=1)
            return torch.sigmoid(self.classifier(x)).squeeze()


    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)

    # Create model
    model = SimpleTransformer(input_dim=n_features)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\n🧠 Transformer Model:")
    print(f"   • Input dimension: {n_features}")
    print(f"   • Hidden dimension: 32")
    print(f"   • Attention heads: 2")
    print(f"   • Parameters: {sum(p.numel() for p in model.parameters())}")

    # Training
    print("\n⚡ Training (5 epochs)...")
    for epoch in range(5):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

        # Accuracy
        with torch.no_grad():
            train_acc = ((outputs > 0.5).float() == y_train_t).float().mean()
            test_outputs = model(X_test_t)
            test_acc = ((test_outputs > 0.5).float() == y_test_t).float().mean()
            test_loss = criterion(test_outputs, y_test_t)

        print(f"   Epoch {epoch + 1}: Loss={loss:.4f}, Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        y_pred_prob = model(X_test_t).numpy()
        y_pred = (y_pred_prob > 0.5).astype(int)

    print(f"\n✅ Final Test Accuracy: {test_acc:.4f}")

# ============================================
# 4. FALLBACK: Simple Classifier (no PyTorch)
# ============================================
else:
    print("\n⚠ Using scikit-learn classifier (no transformer)")

    # Random Forest as fallback
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    accuracy = (y_pred == y_test).mean()
    print(f"\n✅ Random Forest Accuracy: {accuracy:.4f}")

# ============================================
# 5. RESULTS
# ============================================

print("\n" + "=" * 60)
print("📋 RESULTS")
print("=" * 60)

# Calculate metrics
accuracy = (y_pred == y_test).mean()
tp = ((y_pred == 1) & (y_test == 1)).sum()
tn = ((y_pred == 0) & (y_test == 0)).sum()
fp = ((y_pred == 1) & (y_test == 0)).sum()
fn = ((y_pred == 0) & (y_test == 1)).sum()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n🎯 Performance Metrics:")
print(f"   • Accuracy:  {accuracy:.4f}")
print(f"   • Precision: {precision:.4f}")
print(f"   • Recall:    {recall:.4f}")
print(f"   • F1-Score:  {f1:.4f}")

print(f"\n📊 Confusion Matrix:")
print(f"   • True Positives:  {tp}")
print(f"   • True Negatives:  {tn}")
print(f"   • False Positives: {fp}")
print(f"   • False Negatives: {fn}")

# ============================================
# 6. SAMPLE PREDICTIONS
# ============================================

print("\n🔮 Sample Predictions:")
print("-" * 50)
print(f"{'Sample':<10} {'True':<10} {'Predicted':<12} {'Confidence':<10}")
print("-" * 50)

for i in range(min(5, len(y_test))):
    true = "Attack" if y_test[i] == 1 else "Normal"
    pred = "Attack" if y_pred[i] == 1 else "Normal"
    conf = y_pred_prob[i] if y_pred[i] == 1 else 1 - y_pred_prob[i]
    print(f"{i + 1:<10} {true:<10} {pred:<12} {conf:.4f}")

print("\n" + "=" * 60)
print("✅ DONE! Transformer network demonstration complete")
print("=" * 60)