import pickle

import jax
import jax.numpy as jnp

from experiments.logisticRegression.utils import get_dataset, get_tgt_log_density
from variational.exponential_family import GenericNormalDistribution, NormalDistribution
from variational.gaussian_lsvi import gaussian_lsvi

OUTPUT_PATH = "./output"
OP_key = jax.random.PRNGKey(4)
jax.config.update("jax_enable_x64", True)


def experiment(keys, n_samples=100000, n_iter=100, lr_schedule=None, target_residual_schedule=None, title_seq="Seq",
               OUTPUT_PATH="./output", s=""):
    flipped_predictors = get_dataset(dataset="Sonar")
    N, dim = flipped_predictors.shape

    # Gaussian Prior
    my_prior_covariance = 25 * jnp.identity(dim)
    my_prior_covariance = my_prior_covariance.at[0, 0].set(400)
    my_prior_log_density = NormalDistribution(jnp.zeros(dim), my_prior_covariance).log_density
    tgt_log_density = get_tgt_log_density(flipped_predictors, my_prior_log_density)

    # Gaussian Variational Family
    my_variational_family = GenericNormalDistribution(dimension=dim)

    upsilon_init = my_variational_family.get_upsilon(jnp.zeros(dim), jnp.identity(dim))

    PARAMS = {'n_iter': n_iter, 'n_samples': n_samples, 'lr': lr_schedule, 'residual': target_residual_schedule}
    desc = "PIMA dataset, heuristic, full cov. Gaussian, Anon"

    # if not os.path.exists(
    #        f"{OUTPUT_PATH}/heuristic_gaussian_Anon_{n_iter}_{n_samples}_{title_seq}_{OP_key}_u{target_residual_schedule.at[0].get()}.pkl"):
    @jax.vmap
    def f(key):
        res, res_all = gaussian_lsvi(key, tgt_log_density, upsilon_init, n_iter, n_samples, lr_schedule=lr_schedule,
                                     target_residual_schedule=target_residual_schedule,
                                     return_all=False)
        return res, res_all

    res, res_all = f(keys)
    with open(
            f"{OUTPUT_PATH}/heuristic_gaussian_Anon_{n_iter}_{n_samples}_{title_seq}_{OP_key}_{s}.pkl",
            "wb") as f:
        pickle.dump({'desc': desc, 'PARAMS': PARAMS, 'res': res, 'all': res_all}, f)


if __name__ == "__main__":
    n_iter = 100
    Seq_titles = ['sch1', 'sch2']
    interval = jnp.arange(1, n_iter + 1)
    Seq = [jnp.ones(n_iter), 1 / interval]
    Ns = [1e5]
    target_residual_schedules = [jnp.full(n_iter, 10.), jnp.inf]
    n_repetitions = 10
    for key in range(10):
        keys = jax.random.split(jax.random.PRNGKey(key), n_repetitions)
        for idx, title in enumerate(Seq_titles):
            print(title)
            for n_samples in Ns:
                print(n_samples)
                experiment(keys, n_samples=int(n_samples), n_iter=n_iter, lr_schedule=Seq[idx],
                           target_residual_schedule=target_residual_schedules[idx], title_seq=title,
                           OUTPUT_PATH=OUTPUT_PATH, s=key)
