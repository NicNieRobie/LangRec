_IGNORED_CLASSES = set()


def ignore_discovery(cls):
    _IGNORED_CLASSES.add(cls)

    setattr(cls, 'ignore_discovery', True)

    original_init_subclass = cls.__dict__.get('__init_subclass__', None)

    @classmethod
    def _init_subclass(subcls, **kwargs):
        if subcls not in _IGNORED_CLASSES:
            setattr(subcls, 'ignore_discovery', False)

        if original_init_subclass:
            original_init_subclass(subcls, **kwargs)
        else:
            super(cls, subcls).__init_subclass__(**kwargs)

    cls.__init_subclass__ = _init_subclass

    return cls