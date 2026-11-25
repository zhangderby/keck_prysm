from prysm.mathops import np

def abc_psd_2d(samples, dx, a, b, c):
    
    df = 1 / (samples * dx / 1e3) # 1/m
    fx = (df * (np.arange(-samples / 2, samples / 2, 1) + 0.5))
    fx, fy = np.meshgrid(fx, fx)
    fr = np.sqrt(fx ** 2 + fy ** 2)

    psd = a / ((1 + ((fr / b) ** 2)) ** (c / 2.))

    return psd, fr

def von_karman_psd_2d(samples, dx, norm, index, outer_scale, inner_scale, surf_rough):

    df = 1 / (samples * dx / 1e3) # 1/m
    fx = (df * (np.arange(-samples / 2, samples / 2, 1) + 0.5))
    fx, fy = np.meshgrid(fx, fx)
    fr = np.sqrt(fx ** 2 + fy ** 2)

    denom = ((1 / outer_scale) ** 2 + fr ** 2) ** (index / 2)
    psd = ((norm / denom) * np.exp(-((fr / inner_scale) ** 2))) + surf_rough

    return psd, fr

def psd_to_opd(psd, seed=727):

    samples = psd.shape[0]

    rng = np.random.RandomState(seed=seed)
    random_phase = rng.rand(samples, samples) * (2 * np.pi) - np.pi

    psd_noisy = np.sqrt(psd ** 2) * np.exp(1.0j * random_phase)

    opd = samples ** 2 * np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(psd_noisy))).real
    opd -= np.mean(opd)

    return opd