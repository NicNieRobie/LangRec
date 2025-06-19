import hashlib

PREFIX = 'LANGREC'

MODE_NAMES = {
    'finetune': 'FTN',
    'test': 'TST',
    'testtune': 'TTN'
}

DATASET_NAMES = {
    'MOVIELENS': 'MOVIE',
    'STEAM': 'STEAM',
    'GOODREADS': 'GOOD'
}


def get_args_hash(args) -> str:
    args_dict = vars(args)
    sorted_items = sorted(args_dict.items())
    string_repr = repr(sorted_items)

    return hashlib.md5(string_repr.encode()).hexdigest()[:8]


def generate_run_name_and_hash(args):
    args_hash = get_args_hash(args)

    run_name = '_'.join([
        PREFIX,
        args.task,
        DATASET_NAMES[args.dataset],
        args.model,
        MODE_NAMES[args.mode],
        args_hash
    ]).upper()

    return run_name, args_hash
