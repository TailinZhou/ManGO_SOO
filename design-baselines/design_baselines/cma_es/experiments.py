from ray import tune
import click
import ray
import os


@click.group()
def cli():
    """A group of experiments for training Conservative Score Models
    and reproducing our ICLR 2021 results.
    """


#############


# @cli.command()
# @click.option('--local-dir', type=str, default='cma-es-dkitty')
# @click.option('--cpus', type=int, default=24)
# @click.option('--gpus', type=int, default=1)
# @click.option('--num-parallel', type=int, default=1)
# @click.option('--num-samples', type=int, default=1)
# def dkitty(local_dir, cpus, gpus, num_parallel, num_samples):
#     """Evaluate CMA-ES on DKittyMorphology-Exact-v0
#     """

#     # Final Version

#     from design_baselines.cma_es import cma_es
#     ray.init(num_cpus=cpus,
#              num_gpus=gpus,
#              include_dashboard=False,
#              _temp_dir=os.path.expanduser('~/tmp'))
#     tune.run(cma_es, config={
#         "logging_dir": "data",
#         "normalize_ys": True,
#         "normalize_xs": True,
#         "task": "DKittyMorphology-Exact-v0",
#         "task_kwargs": {"relabel": False},
#         "bootstraps": 5,
#         "val_size": 200,
#         "optimize_ground_truth": False,
#         "ensemble_batch_size": 100,
#         "hidden_size": 256,
#         "num_layers": 1,
#         "initial_max_std": 0.2,
#         "initial_min_std": 0.1,
#         "ensemble_lr": 0.001,
#         "ensemble_epochs": 100,
#         "cma_max_iterations": 100,
#         "cma_sigma": 0.5,
#         "solver_samples": 128, "do_evaluation": True},
#         num_samples=num_samples,
#         local_dir=local_dir,
#         resources_per_trial={'cpu': cpus // num_parallel,
#                              'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='cma-es-ant')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def ant(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate CMA-ES on AntMorphology-Exact-v0
    """

    # Final Version

    from design_baselines.cma_es import cma_es
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(cma_es, config={
        "logging_dir": "data",
        "normalize_ys": True,
        "normalize_xs": True,
        "task": "AntMorphology-Exact-v0",
        "task_kwargs": {"relabel": False},
        "bootstraps": 5,
        "val_size": 200,
        "optimize_ground_truth": False,
        "ensemble_batch_size": 100,
        "hidden_size": 256,
        "num_layers": 1,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "ensemble_lr": 0.001,
        "ensemble_epochs": 100,
        "cma_max_iterations": 100,
        "cma_sigma": 0.5,
        "solver_samples": 128, "do_evaluation": True},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='cma-es-hopper')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def hopper(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate CMA-ES on HopperController-Exact-v0
    """

    # Final Version

    from design_baselines.cma_es import cma_es
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(cma_es, config={
        "logging_dir": "data",
        "normalize_ys": True,
        "normalize_xs": True,
        "task": "HopperController-Exact-v0",
        "task_kwargs": {"relabel": False},
        "bootstraps": 5,
        "val_size": 200,
        "optimize_ground_truth": False,
        "ensemble_batch_size": 100,
        "hidden_size": 256,
        "num_layers": 1,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "ensemble_lr": 0.001,
        "ensemble_epochs": 100,
        "cma_max_iterations": 100,
        "cma_sigma": 0.5,
        "solver_samples": 128, "do_evaluation": True},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='cma-es-superconductor')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--oracle', type=str, default="RandomForest")
def superconductor(local_dir, cpus, gpus, num_parallel, num_samples, oracle):
    """Evaluate CMA-ES on Superconductor-RandomForest-v0
    """

    # Final Version

    from design_baselines.cma_es import cma_es
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(cma_es, config={
        "logging_dir": "data",
        "normalize_ys": True,
        "normalize_xs": True,
        "task": f"Superconductor-{oracle}-v0",
        "task_kwargs": {"relabel": False},
        "bootstraps": 5,
        "val_size": 200,
        "optimize_ground_truth": False,
        "ensemble_batch_size": 100,
        "hidden_size": 256,
        "num_layers": 1,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "ensemble_lr": 0.001,
        "ensemble_epochs": 100,
        "cma_max_iterations": 100,
        "cma_sigma": 0.5,
        "solver_samples": 128, "do_evaluation": True},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='cma-es-chembl')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--assay-chembl-id', type=str, default='CHEMBL3885882')
@click.option('--standard-type', type=str, default='MCHC')
def chembl(local_dir, cpus, gpus, num_parallel, num_samples,
           assay_chembl_id, standard_type):
    """Evaluate CMA-ES on ChEMBL-ResNet-v0
    """

    # Final Version

    from design_baselines.cma_es import cma_es
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(cma_es, config={
        "logging_dir": "data",
        "normalize_ys": True,
        "normalize_xs": False,
        "task": f"ChEMBL_{standard_type}_{assay_chembl_id}"
                f"_MorganFingerprint-RandomForest-v0",
        "task_kwargs": {"relabel": False,
                        "dataset_kwargs": dict(
                            assay_chembl_id=assay_chembl_id,
                            standard_type=standard_type)},
        "bootstraps": 5,
        "val_size": 200,
        "optimize_ground_truth": False,
        "use_vae": True,
        "vae_beta": 0.01,
        "vae_epochs": 50,
        "vae_batch_size": 128,
        "vae_hidden_size": 256,
        "vae_latent_size": 256,
        "vae_activation": "relu",
        "vae_kernel_size": 3,
        "vae_num_blocks": 4,
        "vae_lr": 0.0003,
        "ensemble_batch_size": 100,
        "hidden_size": 2048,
        "num_layers": 1,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "ensemble_lr": 0.0003,
        "ensemble_epochs": 50,
        "cma_max_iterations": 100,
        "cma_sigma": 0.5,
        "solver_samples": 128, "do_evaluation": True},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='cma-es-gfp')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--oracle', type=str, default="Transformer")
def gfp(local_dir, cpus, gpus, num_parallel, num_samples, oracle):
    """Evaluate CMA-ES on GFP-Transformer-v0
    """

    # Final Version

    from design_baselines.cma_es import cma_es
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(cma_es, config={
        "logging_dir": "data",
        "normalize_ys": True,
        "normalize_xs": True,
        "task": f"GFP-{oracle}-v0",
        "task_kwargs": {"relabel": False},
        "bootstraps": 5,
        "val_size": 200,
        "optimize_ground_truth": False,
        "use_vae": False,
        "vae_beta": 0.01,
        "vae_epochs": 20,
        "vae_batch_size": 128,
        "vae_hidden_size": 64,
        "vae_latent_size": 256,
        "vae_activation": "relu",
        "vae_kernel_size": 3,
        "vae_num_blocks": 4,
        "vae_lr": 0.001,
        "ensemble_batch_size": 100,
        "hidden_size": 256,
        "num_layers": 1,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "ensemble_lr": 0.001,
        "ensemble_epochs": 100,
        "cma_max_iterations": 100,
        "cma_sigma": 0.5,
        "solver_samples": 128, "do_evaluation": True},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='cma-es-tf-bind-8')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def tf_bind_8(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate CMA-ES on TFBind8-Exact-v0
    """

    # Final Version

    from design_baselines.cma_es import cma_es
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(cma_es, config={
        "logging_dir": "data",
        "normalize_ys": True,
        "normalize_xs": True,
        "task": "TFBind8-Exact-v0",
        "task_kwargs": {"relabel": False},
        "bootstraps": 5,
        "val_size": 200,
        "optimize_ground_truth": False,
        "use_vae": False,
        "vae_beta": 0.01,
        "vae_epochs": 20,
        "vae_batch_size": 128,
        "vae_hidden_size": 64,
        "vae_latent_size": 256,
        "vae_activation": "relu",
        "vae_kernel_size": 3,
        "vae_num_blocks": 3,
        "vae_lr": 0.001,
        "ensemble_batch_size": 100,
        "hidden_size": 256,
        "num_layers": 1,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "ensemble_lr": 0.001,
        "ensemble_epochs": 100,
        "cma_max_iterations": 100,
        "cma_sigma": 0.5,
        "solver_samples": 128, "do_evaluation": True},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='cma-es-tf_bind_10')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def tf_bind_10(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate CMA-ES on TFBind10-Exact-v0
    """

    # Final Version

    from design_baselines.cma_es import cma_es
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(cma_es, config={
        "logging_dir": "data",
        "normalize_ys": True,
        "normalize_xs": True,
        "task": "TFBind10-Exact-v0",
        "task_kwargs": {"relabel": False, "dataset_kwargs": {"max_samples": 10000}},
        "bootstraps": 5,
        "val_size": 200,
        "optimize_ground_truth": False,
        "use_vae": False,
        "vae_beta": 0.01,
        "vae_epochs": 20,
        "vae_batch_size": 128,
        "vae_hidden_size": 64,
        "vae_latent_size": 256,
        "vae_activation": "relu",
        "vae_kernel_size": 3,
        "vae_num_blocks": 4,
        "vae_lr": 0.001,
        "ensemble_batch_size": 100,
        "hidden_size": 256,
        "num_layers": 1,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "ensemble_lr": 0.001,
        "ensemble_epochs": 100,
        "cma_max_iterations": 100,
        "cma_sigma": 0.5,
        "solver_samples": 128, "do_evaluation": True},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='cma-es-nas')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def nas(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate CMA-ES on CIFARNAS-Exact-v0
    """

    # Final Version

    from design_baselines.cma_es import cma_es
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(cma_es, config={
        "logging_dir": "data",
        "normalize_ys": True,
        "normalize_xs": True,
        "task": "CIFARNAS-Exact-v0",
        "task_kwargs": {"relabel": False},
        "bootstraps": 5,
        "val_size": 200,
        "optimize_ground_truth": False,
        "use_vae": False,
        "vae_beta": 0.01,
        "vae_epochs": 20,
        "vae_batch_size": 128,
        "vae_hidden_size": 64,
        "vae_latent_size": 256,
        "vae_activation": "relu",
        "vae_kernel_size": 3,
        "vae_num_blocks": 4,
        "vae_lr": 0.001,
        "ensemble_batch_size": 100,
        "hidden_size": 256,
        "num_layers": 1,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "ensemble_lr": 0.001,
        "ensemble_epochs": 100,
        "cma_max_iterations": 100,
        "cma_sigma": 0.5,
        "solver_samples": 128, "do_evaluation": False},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})

