.. _model_sampling:

#############
MCMC sampling
#############
MDT supports Markov Chain Monte Carlo (MCMC) sampling of all models as a way of recovering the full posterior density of model parameters given the data.
While model fitting provides you only with a maximum likelihood estimate and a standard deviations using the Fisher Information Matrix,
MCMC sampling approximates the full posterior distribution by drawing many samples of the parameters.

In contrary to model fitting, model sampling is currently only available using the Python function :func:`mdt.sample_model`.

The main workflow is similar to the model fitting in that that you load some (pre-processed) MRI data, select a model, and let MDT do the sampling for you.
In most cases you will also want to initialize the sampling routine using a better initialization point, which is typically provided by the model fitting.

The common workflow is to first optimize the model using the model fitting routines and use that as a starting point for the MCMC sampling.
A full example would be:


.. code-block:: python

    import mdt

    input_data = mdt.load_input_data(
        '../b1k_b2k/b1k_b2k_example_slices_24_38',
        '../b1k_b2k/b1k_b2k.prtcl',
        '../b1k_b2k/b1k_b2k_example_slices_24_38_mask')

    inits = mdt.get_optimization_inits('NODDI', input_data, 'output')

    mle = mdt.fit_model('NODDI', input_data, 'output',
                        initialization_data={'inits': inits})

    samples = mdt.sample_model(
        'NODDI',
        input_data,
        'output',
        nmr_samples=1000,
        burnin=0,
        thinning=0,
        initialization_data={'inits': mle}
    )


here, we load the input data, get a suitable optimization starting point, fit the NODDI model and then finally use that as a starting point for the MCMC sampling.


***************
Post-processing
***************
Instead of storing the samples and post-processing the results afterwards, it is also possible to let MDT do some post-processing while it still has the samples in memory.
For example, when you are low on disk space you could do all the post-processing directly after sampling such that you have all your statistic maps without storing all the samples.
Common post-processing options are available using the ``post_processing`` argument to the :func:`mdt.sample_model` function.
More advanced (custom) functionality can be called using the ``post_sampling_cb`` argument.

Common post-processing
======================
Common post-processing, with their defaults are given by:

.. code-block:: python

    mdt.sample_model(
        ...
        post_processing={
            'univariate_ess': False,
            'multivariate_ess': False,
            'maximum_likelihood': False,
            'maximum_a_posteriori': False,
            'average_acceptance_rate': False,
            'model_defined_maps': True,
            'univariate_normal': True}
    )

The individual options are:

* *Univariate ess*: the Effective Sample Size for every parameter
* *Multivariate ess*: a multivariate Effective Sample Size using all parameters
* *Maximum likelihood*: take (per voxel) the samples with the highest log-likelihood
* *Maximum a posteriori*: take (per voxel) the samples with the highest posterior (log-likelihood times log-prior)
* *Average acceptance rate*: compute the average acceptance rate per parameter
* *Model defined maps*: output the additional maps defined in the model definitions
* *Univariate normal*: output a mean and standard deviation per model parameter


Custom post-processing
======================
In addition to the common post-processing options, MDT also allows you to specify a generic callback function for post-processing.
This function takes as input the sampling results and the current model and should output a (nested) dictionary with output maps / folders (when nested).

As an example, to replicate the ``maximum_a_posteriori`` post-processing option we can use:

.. code-block:: python

    def maximum_a_posteriori(sampling_output, composite_model):
        from mdt.utils import results_to_dict

        samples = sampling_output.get_samples()
        log_likelihoods = sampling_output.get_log_likelihoods()
        log_priors = sampling_output.get_log_priors()

        posteriors = log_likelihoods + log_priors

        map_indices = np.argmax(posteriors, axis=1)
        map_samples = samples[range(samples.shape[0]), :, map_indices]

        result_maps = results_to_dict(
            map_samples,
            composite_model.get_free_param_names())

        return {'my_post_processing': result_maps}

    mdt.sample_model(
        ...
        post_sampling_cb=maximum_a_posteriori
    )

this will save all the maps from ``result_maps`` to a sub-directory called ``my_post_processing``.

