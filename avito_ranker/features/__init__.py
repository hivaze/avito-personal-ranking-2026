"""Pre-computed candidate inputs of the ranker.

The ranker computes most of its dense block itself from the raw fields. What
needs other feeds, other users or a separately trained model is computed once
and stored as one float32 array per split:

``item_counts``         occurrences of the item in history snapshots, the same user's later snapshots left out
``covisitation``        co-occurrence of the candidate with the feed's history items in other users' sequences
``feed_context``        how the candidate stands among the other candidates of its feed
``trajectory_encoder``  scores of a self-supervised encoder of the user's history
``exposure``            the capped impression count read by the popularity-light view
``exploration``         per train feed, the probability that it was shown in random order
``finetune_extras``     the 42 extra inputs of the fine-tune stage (val and test only)

Three rules hold for every input. No snapshot of the same user taken after the
feed is read, because it contains the feed's clicks. The feed's own labels are
never read. A train row is built from the information a test row would have,
with one exception: the leave-user-out counts of ``item_counts`` include other
users' snapshots from later days, up to the end of test
(docs/04-validation.md, leak incidents).
"""
from avito_ranker.features.names import (
    ENCODER_INPUTS,
    EXPOSURE_INPUT,
    FEED_CONTEXT_INPUTS,
    FINETUNE_EXTRAS,
    POPULARITY_INPUTS,
    RANKER_INPUTS,
)

__all__ = [
    "ENCODER_INPUTS",
    "EXPOSURE_INPUT",
    "FEED_CONTEXT_INPUTS",
    "FINETUNE_EXTRAS",
    "POPULARITY_INPUTS",
    "RANKER_INPUTS",
]
