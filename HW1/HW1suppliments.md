If you use the `weight_decay` parameter in your optimizer, **remove the manual L2 regularization from your `cal_loss` function**.  
This prevents double regularization.

**How to modify:**

1. **Set weight_decay in optimizer:**
```python
config = {
    # ...existing code...
    'optimizer': 'SGD',
    'optim_hparas': {
        'lr': 0.001,
        'momentum': 0.9,
        'weight_decay': 0.00075  # <-- add this line
    },
    # ...existing code...
}
```

2. **Remove L2 from cal_loss:**
```python
class NeuralNet(nn.Module):
    # ...existing code...
    def cal_loss(self, pred, target):
        ''' Calculate loss (no manual L2 regularization here) '''
        return self.criterion(pred, target)
```

**Summary:**  
- Use **either** manual L2 in `cal_loss` **or** `weight_decay` in the optimizer, **not both**.
- Using `weight_decay` is the standard PyTorch way.


If you use `weight_decay` in the optimizer, **remove the manual L2 regularization from your `cal_loss` function**.  
The loss returned by `model.cal_loss(pred, y)` will then be **just the MSE loss** (no weight decay term included), because PyTorch applies weight decay internally during the optimizer step.

**However, the value you record in `loss_record['train']` will NOT include the weight decay penalty**—it will only be the MSE loss.

---

### **How to record the loss including weight decay?**

You need to **manually add the weight decay term to the loss you record**, even though you don't use it for backward.  
Here’s how:

#### 1. Remove L2 from `cal_loss`:
```python
def cal_loss(self, pred, target):
    return self.criterion(pred, target)
```

#### 2. In your training loop, after `mse_loss = model.cal_loss(pred, y)`, add:
```python
# ...existing code...
mse_loss = model.cal_loss(pred, y)

# Add this block to compute weight decay penalty for logging
weight_decay = config['optim_hparas'].get('weight_decay', 0)
if weight_decay > 0:
    l2_penalty = 0
    for param in model.parameters():
        l2_penalty += torch.sum(param ** 2)
    total_loss = mse_loss + weight_decay * l2_penalty
    loss_record['train'].append(total_loss.detach().cpu().item())
else:
    loss_record['train'].append(mse_loss.detach().cpu().item())
# ...existing code...
```

**Summary:**  
- Use `weight_decay` in optimizer for training.
- For logging, manually add the L2 penalty to the loss you record in `loss_record['train']` if you want to see the "full" loss (MSE + weight decay), just for monitoring.  
- Do NOT add this penalty to the loss you backpropagate (that’s handled by the optimizer).

Let me know if you want the same for validation loss!