from utils.class_library import ClassLibrary

if __name__ == '__main__':
    metrics = ClassLibrary.metrics()

    print({m.name: m for name, m in metrics.class_dict.items()})