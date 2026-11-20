"""What changes when the learning rate cools down, apart from the learning rate.

Val and test feeds were shown to users in random order; most train feeds were
ordered by the production ranker. The last part of the run is spent closer to
the val and test regime in two ways:

* With a small probability (0.02 in the final recipes) a cooldown step trains
  on a batch of exploration feeds instead of the next regular batch. An
  exploration feed is a training feed whose probability of having been shown
  in random order is above 0.5: 164,118 of the 3,151,208 train feeds, or
  134,118 when the 30,000-feed holdout is kept out. The pool is cut into
  batches like the training feeds and reshuffled whenever it is used up.
* The weight of the production-order loss follows the learning rate down,
  from 0.5 at the start of the cooldown to 0.1 when the learning rate reaches
  its floor.
"""
from __future__ import annotations

import numpy as np

from avito_ranker.data.loader import SplitTensors
from avito_ranker.training.optim import Schedule

EXPLORATION_THRESHOLD = 0.5   # a feed belongs to the pool if its exploration probability is above this


def exploration_pool(feeds: np.ndarray, exploration_probability: np.ndarray) -> np.ndarray:
    """The training feeds that were probably shown in random order."""
    return feeds[exploration_probability[feeds] > EXPLORATION_THRESHOLD]


def position_weight(schedule: Schedule, step: int, start: float, end: float) -> float:
    """Weight of the production-order loss at ``step``: ``start`` until the cooldown, then down to ``end``."""
    if not schedule.in_cooldown(step):
        return start
    return end + (start - end) * schedule.decay_left(step)


class ExplorationBatches:
    """Batches of exploration feeds that replace some of the regular batches in the cooldown.

    The draws come from their own generator, seeded with ``seed``, so that
    the regular batch order does not depend on them. :meth:`replaces` is
    called once per cooldown step and nowhere else.
    """

    def __init__(self, split: SplitTensors, pool: np.ndarray, probability: float, seed: int,
                 max_candidates: int, max_feeds: int):
        self.split = split
        self.pool = pool
        self.probability = probability
        self.max_candidates = max_candidates
        self.max_feeds = max_feeds
        self.rng = np.random.default_rng(seed)
        self.plan = self._shuffled_plan()
        self.position = 0
        self.used = 0

    def _shuffled_plan(self) -> list[np.ndarray]:
        return self.split.plan_batches(self.pool, self.max_candidates, self.max_feeds, shuffle=True, rng=self.rng)

    def replaces(self) -> bool:
        """Draw whether this cooldown step trains on exploration feeds."""
        return self.rng.random() < self.probability

    def next(self) -> np.ndarray:
        """The next batch of the pool; the pool is reshuffled when every batch has been used."""
        if self.position >= len(self.plan):
            self.plan, self.position = self._shuffled_plan(), 0
        feeds = self.plan[self.position]
        self.position += 1
        self.used += 1
        return feeds
