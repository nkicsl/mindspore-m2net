# encoding: utf-8

def get_lr_mindspore(lr_init, total_epochs, steps_per_epoch, decay_epochs, gamma=0.1, warmup_epoch=0, warmup_factor=0.5):
    """ Stepped learning rate scheduler """
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    if isinstance(decay_epochs, str):
        decay_epochs = list(map(int, decay_epochs.split(',')))
    elif isinstance(decay_epochs, (int, float)):
        decay_epochs = [decay_epochs]

    decay_epochs = decay_epochs.copy()
    mult = 1

    for i in range(total_steps):
        ep = np.floor(1. * i / steps_per_epoch)  # + 1
        warmup_mult = 1

        if ep < warmup_epoch:
            alpha = ep / warmup_epoch
            warmup_mult = warmup_factor * (1 - alpha) + alpha

        if ep in decay_epochs:
            mult *= gamma
            decay_epochs.remove(ep)

        lr = lr_init * mult * warmup_mult

        lr_each_step.append(lr)
    lr_each_step = np.array(lr_each_step, dtype=np.float32)
    return lr_each_step    