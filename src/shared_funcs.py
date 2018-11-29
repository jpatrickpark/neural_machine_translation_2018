import config

def reference_unk_replace(batch, trg, phase_iter):
    reference = []
    for each in batch.idx:
        reference.append(" ".join(['<unk>' if trg.vocab.stoi[s] == 0 else s for s in phase_iter.dataset[each].trg]))
    return reference

def pad(l, max_length):
    while len(l) < max_length + 2:
        l.append(config.PAD_TOKEN)
    return l
