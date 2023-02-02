import copy

import numpy as np
from scipy.special import logsumexp

from ...core.prior import Interped
from ...core.utils.log import logger
from ..utils import noise_weighted_inner_product
from .base import GravitationalWaveTransient
from .relative import RelativeBinningGravitationalWaveTransient
from .roq import ROQGravitationalWaveTransient


class PolarizationMarginalizedGWT(GravitationalWaveTransient):

    def __init__(
        self,
        interferometers,
        waveform_generator,
        distance_marginalization=False,
        phase_marginalization=False,
        priors=None,
        distance_marginalization_lookup_table=None,
        reference_frame="sky",
        time_reference="geocenter",
    ):
        super(PolarizationMarginalizedGWT, self).__init__(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            time_marginalization=False,
            distance_marginalization=distance_marginalization,
            phase_marginalization=phase_marginalization,
            calibration_marginalization=False,
            priors=priors,
            distance_marginalization_lookup_table=distance_marginalization_lookup_table,
            jitter_time=False,
            time_reference=time_reference,
            reference_frame=reference_frame,
        )
        self._check_marginalized_prior_is_set(key="psi")
        self.pols = np.linspace(0, np.pi, 101)[:-1]
        self._pol_response = np.exp(-2j * self.pols)
        self.polarization_prior = self.priors["psi"].prob(self.pols)
        self.polarization_prior /= np.trapz(self.polarization_prior, self.pols)
        self.polarization_prior /= len(self.pols)
        self.polarization_marginalization = True
        priors["psi"] = 0.0
        self._marginalized_parameters.append("psi")

    def calculate_snrs(self, waveform_polarizations, interferometer):
        parameters = self.parameters
        response_parameters = dict(
            ra=parameters["ra"],
            dec=parameters["dec"],
            time=parameters["geocent_time"],
            psi=parameters["psi"],
        )
        responses = (
            interferometer.antenna_response(**response_parameters, mode="plus")
            + 1j * interferometer.antenna_response(**response_parameters, mode="cross")
        ) * self._pol_response

        time_shift = interferometer.time_delay_from_geocenter(
            parameters['ra'], parameters['dec'], parameters['geocent_time']
        )

        mask = interferometer.strain_data.frequency_mask
        frequencies = interferometer.strain_data.frequency_array

        # Be careful to first subtract the two GPS times which are ~1e9 sec.
        # And then add the time_shift which varies at ~1e-5 sec
        dt_geocent = parameters['geocent_time'] - interferometer.strain_data.start_time
        dt = dt_geocent + time_shift
        time_shift = np.exp(-1j * 2 * np.pi * dt * frequencies[mask])

        calibration = interferometer.calibration_model.get_calibration_factor(
            frequencies[mask],
            prefix=f'recalib_{interferometer.name}_',
            **parameters
        )

        d_inner_h = dict()
        h_inner_h = dict()
        for mode in ["plus", "cross"]:
            signal = waveform_polarizations[mode].copy()
            signal *= mask
            signal[mask] *= time_shift * calibration
            d_inner_h[mode] = interferometer.inner_product(signal)
            h_inner_h[mode] = interferometer.optimal_snr_squared(signal)
        h_inner_h["pc"] = noise_weighted_inner_product(
            waveform_polarizations["plus"][mask],
            waveform_polarizations["cross"][mask],
            interferometer.power_spectral_density_array[mask],
            interferometer.duration,
        )

        d_inner_h_array = d_inner_h["plus"] * responses.real + d_inner_h["cross"] * responses.imag
        optimal_snr_squared_array = (
            h_inner_h["plus"] * responses.real**2
            + h_inner_h["cross"] * responses.imag**2
            + 2 * np.real(h_inner_h["pc"]) * responses.real * responses.imag
        )
        d_inner_h = d_inner_h_array[0]
        optimal_snr_squared = abs(optimal_snr_squared_array)[0]
        complex_matched_filter_snr = d_inner_h / optimal_snr_squared**0.5

        return self._CalculatedSNRs(
            d_inner_h=d_inner_h, optimal_snr_squared=optimal_snr_squared,
            complex_matched_filter_snr=complex_matched_filter_snr,
            d_inner_h_array=d_inner_h_array,
            optimal_snr_squared_array=optimal_snr_squared_array.real,
        )

    def compute_log_likelihood_from_snrs(self, total_snrs):
        if self.distance_marginalization:
            log_l = self.distance_marginalized_likelihood(
                d_inner_h=total_snrs.d_inner_h_array, h_inner_h=total_snrs.optimal_snr_squared_array)

        elif self.phase_marginalization:
            log_l = self.phase_marginalized_likelihood(
                d_inner_h=total_snrs.d_inner_h_array, h_inner_h=total_snrs.optimal_snr_squared_array)

        else:
            log_l = np.real(total_snrs.d_inner_h_array) - total_snrs.optimal_snr_squared_array / 2

        log_l = logsumexp(log_l.real, b=self.polarization_prior)

        return log_l

    def generate_posterior_sample_from_marginalized_likelihood(self):
        """
        Reconstruct the distance posterior from a run which used a likelihood
        which explicitly marginalised over time/distance/phase.

        See Eq. (C29-C32) of https://arxiv.org/abs/1809.02293

        Returns
        =======
        sample: dict
            Returns the parameters with new samples.

        Notes
        =====
        This involves a deepcopy of the signal to avoid issues with waveform
        caching, as the signal is overwritten in place.
        """
        if len(self._marginalized_parameters) > 0:
            signal_polarizations = copy.deepcopy(
                self.waveform_generator.frequency_domain_strain(self.parameters)
            )
        else:
            return self.parameters

        new_psi = self.generate_polarization_sample_from_marginalized_likelihood(
            signal_polarizations=signal_polarizations
        )
        self.parameters["psi"] = new_psi
        if self.distance_marginalization:
            new_distance = self.generate_distance_sample_from_marginalized_likelihood(
                signal_polarizations=signal_polarizations)
            self.parameters['luminosity_distance'] = new_distance
        if self.phase_marginalization:
            new_phase = self.generate_phase_sample_from_marginalized_likelihood(
                signal_polarizations=signal_polarizations)
            self.parameters['phase'] = new_phase
        return self.parameters.copy()

    def generate_polarization_sample_from_marginalized_likelihood(
            self, signal_polarizations=None):
        """
        Generate a single sample from the posterior distribution for polarization
        when using a likelihood which explicitly marginalises over
        distance.

        Parameters
        ==========
        signal_polarizations: dict, optional
            Polarizations modes of the template.

        Returns
        =======
        new_distance: float
            Sample from the distance posterior.
        """
        self.parameters.update(self.get_sky_frame_parameters())
        if signal_polarizations is None:
            signal_polarizations = \
                self.waveform_generator.frequency_domain_strain(self.parameters)

        d_inner_h = np.zeros(len(self.pols), dtype=complex)
        h_inner_h = np.zeros(len(self.pols))

        for interferometer in self.interferometers:
            snrs = self.calculate_snrs(
                waveform_polarizations=signal_polarizations,
                interferometer=interferometer,
            )

            d_inner_h += snrs.d_inner_h_array
            h_inner_h += snrs.optimal_snr_squared_array

        if self.distance_marginalization:
            log_like = self.distance_marginalized_likelihood(d_inner_h, h_inner_h)
        elif self.phase_marginalization:
            log_like = self.phase_marginalized_likelihood(d_inner_h, h_inner_h)
        else:
            log_like = (d_inner_h.real - h_inner_h.real / 2)

        post = np.exp(log_like - max(log_like)) * self.polarization_prior
        post = np.concatenate([post, [post[0]]])
        pols = np.concatenate([self.pols, [self.pols[0]]])

        new_distance = Interped(pols, post).sample()

        return new_distance


class RelBinPolarizationMarginalizedGWT(RelativeBinningGravitationalWaveTransient):

    def __init__(
        self,
        interferometers,
        waveform_generator,
        fiducial_parameters=None,
        parameter_bounds=None,
        maximization_kwargs=None,
        update_fiducial_parameters=False,
        distance_marginalization=False,
        phase_marginalization=False,
        priors=None,
        distance_marginalization_lookup_table=None,
        reference_frame="sky",
        time_reference="geocenter",
        chi=1,
        epsilon=0.5
    ):
        self.per_detector_fiducial_waveform_modes = dict()
        self.pols = np.linspace(0, np.pi, 101)[:-1]
        self._pol_response = np.exp(-2j * self.pols)
        self.polarization_prior = priors["psi"].prob(self.pols)
        self.polarization_prior /= np.trapz(self.polarization_prior, self.pols)
        self.polarization_prior /= len(self.pols)
        self.polarization_marginalization = True
        super(RelBinPolarizationMarginalizedGWT, self).__init__(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            fiducial_parameters=fiducial_parameters,
            parameter_bounds=parameter_bounds,
            maximization_kwargs=maximization_kwargs,
            update_fiducial_parameters=update_fiducial_parameters,
            time_marginalization=False,
            distance_marginalization=distance_marginalization,
            phase_marginalization=phase_marginalization,
            priors=priors,
            distance_marginalization_lookup_table=distance_marginalization_lookup_table,
            jitter_time=False,
            time_reference=time_reference,
            reference_frame=reference_frame,
            # polarization_marginalization=True,
            chi=chi,
            epsilon=epsilon,
        )
        self._marginalized_parameters.append("psi")
        self._check_marginalized_prior_is_set(key="psi")
        priors["psi"] = 0.0

    def setup_bins(self):
        super(RelBinPolarizationMarginalizedGWT, self).setup_bins()
        for interferometer in self.interferometers:
            self.per_detector_fiducial_waveform_modes[interferometer.name] = self.mode_by_mode_response(
                waveform_polarizations={key: self.fiducial_polarizations[key][self.bin_inds] for key in ["plus", "cross"]},
                parameters=self.fiducial_parameters,
                interferometer=interferometer,
            )

    def compute_summary_data(self):
        summary_data = dict()

        for interferometer in self.interferometers:
            summary_data[interferometer.name] = dict()
            for mode in ["plus", "cross"]:
                mask = interferometer.frequency_mask
                masked_frequency_array = interferometer.frequency_array[mask]
                masked_bin_inds = []
                for edge in self.bin_freqs:
                    index = np.where(masked_frequency_array == edge)[0][0]
                    masked_bin_inds.append(index)
                masked_strain = interferometer.frequency_domain_strain[mask]
                masked_h0 = self.fiducial_polarizations[mode][mask]
                # masked_h0 = self.per_detector_fiducial_waveforms[interferometer.name][mask]
                masked_psd = interferometer.power_spectral_density_array[mask]
                duration = interferometer.duration
                a0, b0, a1, b1 = np.zeros((4, self.number_of_bins), dtype=complex)

                for i in range(self.number_of_bins):
                    start_idx = masked_bin_inds[i]
                    end_idx = masked_bin_inds[i + 1]
                    start = masked_frequency_array[start_idx]
                    stop = masked_frequency_array[end_idx]
                    idxs = slice(start_idx, end_idx)

                    strain = masked_strain[idxs]
                    h0 = masked_h0[idxs]
                    psd = masked_psd[idxs]

                    frequencies = masked_frequency_array[idxs]
                    central_frequency = (start + stop) / 2
                    delta_frequency = frequencies - central_frequency

                    a0[i] = noise_weighted_inner_product(h0, strain, psd, duration)
                    b0[i] = noise_weighted_inner_product(h0, h0, psd, duration)
                    a1[i] = noise_weighted_inner_product(h0, strain * delta_frequency, psd, duration)
                    b1[i] = noise_weighted_inner_product(h0, h0 * delta_frequency, psd, duration)

                summary_data[interferometer.name][mode] = (a0, a1, b0, b1)

        self.summary_data = summary_data

    def mode_by_mode_response(self, waveform_polarizations, parameters, interferometer):
        parameters = parameters.copy()
        parameters.update(self.get_sky_frame_parameters(parameters=parameters))
        time_shift = interferometer.time_delay_from_geocenter(
            parameters['ra'], parameters['dec'], parameters['geocent_time']
        )
        frequencies = self.bin_freqs
        dt_geocent = parameters['geocent_time'] - interferometer.strain_data.start_time
        dt = dt_geocent + time_shift
        time_shift = np.exp(-1j * 2 * np.pi * dt * frequencies)

        calibration = interferometer.calibration_model.get_calibration_factor(
            frequencies,
            prefix=f'recalib_{interferometer.name}_',
            **parameters
        )
        strain = dict()
        for mode in ["plus", "cross"]:
            signal = waveform_polarizations[mode].copy()
            signal *= time_shift * calibration
            strain[mode] = signal
        return strain

    def compute_waveform_ratio_per_interferometer(self, waveform_polarizations, interferometer):
        parameters = self.parameters
        strain = self.mode_by_mode_response(
            waveform_polarizations=waveform_polarizations,
            parameters=parameters,
            interferometer=interferometer,
        )
        ratios = dict()
        for mode in ["plus", "cross"]:
            reference_strain = self.per_detector_fiducial_waveform_modes[interferometer.name][mode]
            waveform_ratio = strain[mode] / reference_strain
            r0 = (waveform_ratio[1:] + waveform_ratio[:-1]) / 2
            r1 = (waveform_ratio[1:] - waveform_ratio[:-1]) / self.bin_widths
            ratios[mode] = [r0, r1]
        return ratios

    def calculate_snrs(self, waveform_polarizations, interferometer):
        ratios = self.compute_waveform_ratio_per_interferometer(
            waveform_polarizations=waveform_polarizations,
            interferometer=interferometer,
        )
        summary = self.summary_data[interferometer.name]
        # a0, a1, b0, b1 = self.summary_data[interferometer.name]

        d_inner_h = dict()
        h_inner_h = dict()
        for mode in ["plus", "cross"]:
            a0, a1, b0, b1 = summary[mode]
            r0, r1 = ratios[mode]
            d_inner_h[mode] = np.sum(a0 * np.conjugate(r0) + a1 * np.conjugate(r1))
            h_inner_h[mode] = np.sum(b0 * np.abs(r0) ** 2 + 2 * b1 * np.real(r0 * np.conjugate(r1)))
        h_inner_h["pc"] = np.sum(np.real(
            summary["plus"][2] * np.abs(ratios["cross"][0])**2
            + 2 * summary["plus"][3] * np.real(ratios["cross"][0] * np.conjugate(ratios["cross"][1]))
            + summary["cross"][2] * np.abs(ratios["plus"][0])**2
            + 2 * summary["cross"][3] * np.real(ratios["plus"][0] * np.conjugate(ratios["plus"][1]))
        ))

        parameters = self.parameters
        response_parameters = dict(
            ra=parameters["ra"],
            dec=parameters["dec"],
            time=parameters["geocent_time"],
            psi=parameters["psi"],
        )
        responses = (
            interferometer.antenna_response(**response_parameters, mode="plus")
            + 1j * interferometer.antenna_response(**response_parameters, mode="cross")
        ) * self._pol_response
        d_inner_h_array = d_inner_h["plus"] * responses.real + d_inner_h["cross"] * responses.imag
        optimal_snr_squared_array = (
            h_inner_h["plus"] * responses.real**2
            + h_inner_h["cross"] * responses.imag**2
            + 0 * np.real(h_inner_h["pc"]) * responses.real * responses.imag
        )
        d_inner_h = d_inner_h_array[0]
        optimal_snr_squared = abs(optimal_snr_squared_array)[0]
        complex_matched_filter_snr = d_inner_h / optimal_snr_squared**0.5

        return self._CalculatedSNRs(
            d_inner_h=d_inner_h, optimal_snr_squared=optimal_snr_squared,
            complex_matched_filter_snr=complex_matched_filter_snr,
            d_inner_h_array=d_inner_h_array,
            optimal_snr_squared_array=optimal_snr_squared_array.real,
        )

    def compute_log_likelihood_from_snrs(self, total_snrs):
        return PolarizationMarginalizedGWT.compute_log_likelihood_from_snrs(self, total_snrs=total_snrs)

    def generate_polarization_sample_from_marginalized_likelihood(
            self, signal_polarizations=None):
        return PolarizationMarginalizedGWT.generate_polarization_sample_from_marginalized_likelihood(
            self=self, signal_polarizations=signal_polarizations
        )

    def generate_posterior_sample_from_marginalized_likelihood(self):
        return PolarizationMarginalizedGWT.generate_posterior_sample_from_marginalized_likelihood(self)


class ROQPolarizationMarginalizedGWT(ROQGravitationalWaveTransient):
    def __init__(
        self,
        interferometers,
        waveform_generator,
        weights=None,
        linear_matrix=None,
        quadratic_matrix=None,
        roq_params=None,
        roq_params_check=True,
        roq_scale_factor=1,
        distance_marginalization=False,
        phase_marginalization=False,
        priors=None,
        distance_marginalization_lookup_table=None,
        reference_frame="sky",
        time_reference="geocenter",
    ):
        super(ROQPolarizationMarginalizedGWT, self).__init__(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            time_marginalization=False,
            distance_marginalization=distance_marginalization,
            phase_marginalization=phase_marginalization,
            priors=priors,
            distance_marginalization_lookup_table=distance_marginalization_lookup_table,
            jitter_time=False,
            time_reference=time_reference,
            reference_frame=reference_frame,
            weights=weights,
            linear_matrix=linear_matrix,
            quadratic_matrix=quadratic_matrix,
            roq_params=roq_params,
            roq_scale_factor=roq_scale_factor,
            roq_params_check=roq_params_check,
        )
        self._check_marginalized_prior_is_set(key="psi")
        self.pols = np.linspace(0, np.pi, 101)[:-1]
        self._pol_response = np.exp(-2j * self.pols)
        self.polarization_prior = self.priors["psi"].prob(self.pols)
        self.polarization_prior /= np.trapz(self.polarization_prior, self.pols)
        self.polarization_prior /= len(self.pols)
        self.polarization_marginalization = True
        priors["psi"] = 0.0
        self._marginalized_parameters.append("psi")

    def generate_polarization_sample_from_marginalized_likelihood(
            self, signal_polarizations=None):
        return PolarizationMarginalizedGWT.generate_polarization_sample_from_marginalized_likelihood(
            self=self, signal_polarizations=signal_polarizations
        )

    def generate_posterior_sample_from_marginalized_likelihood(self):
        return PolarizationMarginalizedGWT.generate_posterior_sample_from_marginalized_likelihood(self)

    def compute_log_likelihood_from_snrs(self, total_snrs):
        return PolarizationMarginalizedGWT.compute_log_likelihood_from_snrs(total_snrs=total_snrs)

    def calculate_snrs(self, waveform_polarizations, interferometer):
        parameters = self.parameters
        response_parameters = dict(
            ra=parameters["ra"],
            dec=parameters["dec"],
            time=parameters["geocent_time"],
            psi=parameters["psi"],
        )
        responses = (
            interferometer.antenna_response(**response_parameters, mode="plus")
            + 1j * interferometer.antenna_response(**response_parameters, mode="cross")
        ) * self._pol_response

        time_shift = interferometer.time_delay_from_geocenter(
            parameters['ra'], parameters['dec'], parameters['geocent_time']
        )

        # Be careful to first subtract the two GPS times which are ~1e9 sec.
        # And then add the time_shift which varies at ~1e-5 sec
        dt_geocent = parameters['geocent_time'] - interferometer.strain_data.start_time
        ifo_time = dt_geocent + time_shift
        indices, in_bounds = self._closest_time_indices(
            ifo_time, self.weights['time_samples']
        )
        if not in_bounds:
            logger.debug("SNR calculation error: requested time at edge of ROQ time samples")
            return self._CalculatedSNRs(
                d_inner_h=np.nan_to_num(-np.inf),
                optimal_snr_squared=0,
                complex_matched_filter_snr=np.nan_to_num(-np.inf),
            )

        calibration_linear = interferometer.calibration_model.get_calibration_factor(
            self.frequency_nodes_linear,
            prefix=f'recalib_{interferometer.name}_',
            **parameters
        )
        calibration_quadratic = interferometer.calibration_model.get_calibration_factor(
            self.frequency_nodes_quadratic,
            prefix=f'recalib_{interferometer.name}_',
            **parameters
        )

        d_inner_h = dict()
        h_inner_h = dict()
        for mode in ["plus", "cross"]:
            linear = waveform_polarizations["linear"][mode] * calibration_linear
            quadratic = waveform_polarizations["quadratic"][mode] * calibration_quadratic
            d_inner_h_tc_array = np.einsum(
                'i,ji->j',
                np.conjugate(linear),
                self.weights[interferometer.name + '_linear'][indices]
            )

            d_inner_h[mode] = self._interp_five_samples(
                self.weights['time_samples'][indices], d_inner_h_tc_array, ifo_time
            )
            h_inner_h[mode] = np.vdot(
                np.abs(quadratic)**2,
                self.weights[interferometer.name + '_quadratic']
            )

        h_inner_h["pc"] = np.vdot(
            waveform_polarizations["quadratic"]["plus"]
            * waveform_polarizations["quadratic"]["cross"].conjugate()
            * abs(calibration_quadratic)**2,
            self.weights[interferometer.name + '_quadratic']
        )

        d_inner_h_array = d_inner_h["plus"] * responses.real + d_inner_h["cross"] * responses.imag
        optimal_snr_squared_array = (
            h_inner_h["plus"] * responses.real**2
            + h_inner_h["cross"] * responses.imag**2
            + 2 * np.real(h_inner_h["pc"]) * responses.real * responses.imag
        )
        d_inner_h = d_inner_h_array[0]
        optimal_snr_squared = abs(optimal_snr_squared_array)[0]
        complex_matched_filter_snr = d_inner_h / optimal_snr_squared**0.5

        return self._CalculatedSNRs(
            d_inner_h=d_inner_h, optimal_snr_squared=optimal_snr_squared,
            complex_matched_filter_snr=complex_matched_filter_snr,
            d_inner_h_array=d_inner_h_array,
            optimal_snr_squared_array=optimal_snr_squared_array.real,
        )
