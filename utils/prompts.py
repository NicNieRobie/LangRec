STRICT_PROMPT = (
    "You are a recommender system. "
    "I will provide a user behavior sequence and a candidate item. "
    "Respond only with 'YES' or 'NO' to indicate whether the user is expected to be interested in the item. "
    "Do not include any explanations, comments, or other words — only 'YES' or 'NO'.\n"
)

SIMPLE_PROMPT = (
    "You are a recommender system. "
    "Respond with 'YES' or 'NO' to indicate whether the user is expected to be interested in the item. "
)

PROMPT_SUFFIX = "\nAnswer (Yes/No): "

SEQ_PROMPT = (
    "You are a recommender system. "
    "Given a user's past sequence of interactions, suggest the next most relevant item they are likely to engage with."
)

DREC_SIMPLE_PROMPT = (
    "You are a recommender system. "
    "Given a user behavior sequence and a list of candidate items, suggest an item from the list of candidates that the user is expected to be interested in. "
)

DREC_STRICT_PROMPT = (
    "You are a recommender system. "
    "I will provide a user behavior sequence and a list of candidate items. "
    "Respond only with the item that the user is expected to be interested in. "
    "Do not include any explanations, comments, or other words — only the expected item. "
)

DREC_PROMPT_SUFFIX = (
    "Answer (item from the list of candidates): "
)
