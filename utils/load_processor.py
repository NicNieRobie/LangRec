from utils.discovery.class_library import ClassLibrary


def load_processor(dataset, task, data_path=None):
    assert task in ['ctr', 'seq'], f'Task {task} is not supported'

    processors = ClassLibrary.processors(task)

    if dataset not in processors:
        raise ValueError(f'Unknown dataset: {dataset}')

    processor = processors[dataset]

    return processor(data_path=data_path)
