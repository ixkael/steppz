import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
from affine import affine_sample

class CatalogSimulator:

    def __init__(self, sps_prior, starting_state, flux_models, data_model, selection_model, additional_prior):
        self.current_state = starting_state
        self.n_walkers = len(starting_state[0])
        self.sps_prior = sps_prior
        self.flux_models = flux_models
        self.data_model = data_model
        self.selection_model = selection_model
        self.additional_prior = additional_prior

        # declare any transforms that occur on the physical SPS parameters before passing to Speculator models
        self.transform = tfb.Blockwise([tfb.Identity() for _ in range(self.sps_prior.n_sps_parameters - 1)])

    def sample_selected_objects(
        self, n_steps, prior_hyperparameters, datamodel_hyperparameters
    ):
        # sample parameters from prior
        def log_prob(parameters, *args):
            # annoying that the model magnitudes need to be calculated twice...
            # could probably change the structure a little bit to avoid that?
            prior_lnp = self.sps_prior.log_prob(parameters, *args)
            model_magnitudes = self.flux_models.magnitudes(self.transform(parameters[..., 1:]), parameters[..., 0])
            return prior_lnp + self.additional_prior(parameters, model_magnitudes)

        parameters = affine_sample(
            log_prob, n_steps, self.current_state, args=prior_hyperparameters, progressbar=False
        )
        # Update state
        self.current_state = [
            parameters[-1, 0:self.n_walkers,:],
            parameters[-1, self.n_walkers:2*self.n_walkers, :]
        ]

        model_fluxes = self.flux_models.fluxes(self.transform(parameters[..., 1:]), parameters[..., 0])
        # TODO: not reshape, and keep separation between walkers?
        parameters = parameters.numpy().reshape(-1, parameters.shape[-1])
        model_fluxes = model_fluxes.numpy().reshape(-1, model_fluxes.shape[-1])
        # use data model to generate and add noise, apply zeropoints
        corrected_noisy_fluxes, corrected_flux_sigmas = self.data_model(
            model_fluxes, datamodel_hyperparameters
        )
        # use selection model to figure out which galaxies are kept
        selected = self.selection_model(
            corrected_noisy_fluxes, corrected_flux_sigmas
        )
        # return selected objects (if any!)
        n_kept = np.sum(selected)
        return n_kept, parameters[selected, :],\
                corrected_noisy_fluxes[selected, :], corrected_flux_sigmas[selected, :]

    def generate_catalog(
        self, n_obj_target, prior_hyperparameters,
        datamodel_hyperparameters, n_steps=10, n_iterations_burnin=5, verbose=True
    ):
        n_kept_total = 0

        all_parameters, all_fluxes, all_flux_sigmas = [], [], []
        n_iterations = 0
        while n_kept_total < n_obj_target:
            n_kept, parameters, fluxes, flux_sigmas =\
                self.sample_selected_objects(
                    n_steps, prior_hyperparameters, datamodel_hyperparameters
                )
            n_iterations += 1
            if n_iterations > n_iterations_burnin: # burnin phase
                all_parameters.append(parameters)
                all_fluxes.append(fluxes)
                all_flux_sigmas.append(flux_sigmas)
                n_kept_total += np.sum(n_kept)
            if verbose:
                print('Iteration', n_iterations, 'kept', n_kept, 'objects (', n_kept_total, 'kept in total)')
        return np.vstack(all_parameters), np.vstack(all_fluxes), np.vstack(all_flux_sigmas)
