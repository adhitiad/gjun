import asyncio
import json
import random

import ray
import redis.asyncio as redis
from deap import algorithms, base, creator, tools

from config import settings
from logging_config import setup_logger

# Coba import Rust
try:
    import turbo_math

    RUST_AVAILABLE = True
except:
    RUST_AVAILABLE = False

logger = setup_logger("Meta-Optimizer")
ray.init(ignore_reinit_error=True)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


@ray.remote
def eval_remote(ind):
    # Simulasi Backtest (Placeholder logic)
    score = (ind[0] * 1.0) + (ind[1] * 1.5)
    if RUST_AVAILABLE:
        # Dummy data for rust
        dummy_prices = [1.0, 1.1, 1.2, 1.1]
        dummy_sigs = [1, 0, 2, 0]
        score += turbo_math.fast_backtest(dummy_prices, dummy_sigs, 0.001) / 10000.0
    return (score,)


class GeneticLab:
    def __init__(self):
        kwargs = {
            "host": settings.REDIS_HOST,
            "port": settings.REDIS_PORT,
            "decode_responses": True,
        }
        if settings.REDIS_PASSWORD:
            kwargs["password"] = settings.REDIS_PASSWORD
        self.r = redis.Redis(**kwargs)
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", random.uniform, 0.1, 2.0)
        self.toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            self.toolbox.attr_float,
            n=6,
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    async def evolve(self):
        pop = self.toolbox.population(n=50)
        for g in range(5):
            offspring = list(
                map(self.toolbox.clone, self.toolbox.select(pop, len(pop)))
            )
            offspring = algorithms.varAnd(offspring, self.toolbox, 0.5, 0.2)

            # Distributed Eval
            futures = [
                eval_remote.remote(ind) for ind in offspring if not ind.fitness.valid
            ]
            results = ray.get(futures)

            for ind, fit in zip(
                [ind for ind in offspring if not ind.fitness.valid], results
            ):
                ind.fitness.values = fit
            pop[:] = offspring

        best = tools.selBest(pop, 1)[0]
        dna = {"w_tech": best[0], "w_rl": best[1], "thresh_buy": best[4]}
        await self.r.set("system:dna", json.dumps(dna))
        logger.info(f"🧬 New DNA: {dna}")

    async def run(self):
        while True:
            await self.evolve()
            await asyncio.sleep(21600)


if __name__ == "__main__":
    asyncio.run(GeneticLab().run())
