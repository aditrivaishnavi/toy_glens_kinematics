
"""train_integration_snippet.py

Example integration in training loop.
"""

import torch
from scheduled_mask import ScheduledCoreMask

SCHEDULE = [
    (0, 7, 0.7),
    (10, 5, 0.5),
    (30, 3, 0.3),
]

core_mask = ScheduledCoreMask(SCHEDULE, image_size=64, fill_value=0.0)

for epoch in range(num_epochs):
    radius, prob = core_mask.get_current_params(epoch)
    logger.info(f"epoch={epoch} core_mask radius={radius} prob={prob}")

    model.train()
    for batch in train_loader:
        imgs, labels = batch["img"], batch["y"]
        imgs = core_mask(imgs, epoch, deterministic=False)
        logits = model(imgs)
        loss = criterion(logits, labels)
        ...
