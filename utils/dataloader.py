def get_steps(dataloader):
    return (len(dataloader.dataset) + dataloader.batch_size - 1) // dataloader.batch_size