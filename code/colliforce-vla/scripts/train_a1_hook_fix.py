# Quick fix: change line 105 from .language_model.model.layers to .language_model.layers
import re
with open('./scripts/train_a1_hook.py', 'r') as f:
    content = f.read()
content = content.replace(
    'model.paligemma_with_expert.paligemma.language_model.model.layers',
    'model.paligemma_with_expert.paligemma.language_model.layers'
)
with open('./scripts/train_a1_hook.py', 'w') as f:
    f.write(content)
print("FIXED")
