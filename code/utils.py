import numpy as np
import tensorflow as tf
from speculator import Speculator
import tensorflow_probability as tfp
tfd = tfp.distributions

# natural log of 10 and ln(2pi)/2
ln10_ = tf.constant(np.log(10), dtype=tf.float32)
halfln2pi_ = tf.constant(0.5*np.log(2*np.pi), dtype=tf.float32)

# set up distance modulus (fitting function) parameters
Om_ = tf.constant(0.286, dtype=tf.float32) # omega matter
H0_ = tf.constant(69.32, dtype=tf.float32) # Hubble constant
c_ = tf.constant(299792.458, dtype=tf.float32) # speed of light
A0_ = tf.constant(c_/H0_, dtype=tf.float32)
s_ = tf.constant(((1-Om_)/Om_)**(1./3.), dtype=tf.float32)
B0_ = tf.constant(2*np.sqrt((1-Om_)/Om_ +1), dtype=tf.float32)
B1_ = tf.constant(-0.154*s_, dtype=tf.float32)
B2_ = tf.constant(0.4304*s_**2, dtype=tf.float32)
B3_ = tf.constant(0.19097*s_**3, dtype=tf.float32)
B4_ = tf.constant(0.066941*s_**4, dtype=tf.float32)
eta0_ = tf.constant(B0_*(1 + B1_ + B2_ + B3_ + B4_)**(-0.125), dtype=tf.float32)


def flux2mag(fluxes, mag_zeropoint=22.5):
    """
    Flux to magnitude convertion (arbitrary array shapes)
    """
    mags = -2.5 * tf.math.log(fluxes) / tf.math.log(10.0) + mag_zeropoint
    return mags


def mag2flux(mags, mag_zeropoint=22.5):
    """
    Magnitude to flux convertion, with error (arbitrary array shapes)
    """
    fluxes = 10 ** (-0.4 * (mags - mag_zeropoint))
    return fluxes


def flux2mag_witherr(fluxes, fluxerrs, mag_zeropoint=22.5):
    """
    Flux to magnitude convertion (arbitrary array shapes)
    """
    mags = -2.5 * tf.math.log(fluxes) / tf.math.log(10.0) + mag_zeropoint
    magerrs = fluxerrs / fluxes * 2.5 / tf.math.log(10.0)
    return mags, magerrs


def mag2flux_witherr(mags, magerrs, mag_zeropoint=22.5):
    """
    Magnitude to flux convertion, with error (arbitrary array shapes)
    """
    fluxes = 10 ** (-0.4 * (mags - mag_zeropoint))
    fluxerrs = fluxes * (10 ** tf.math.abs(0.4 * magerrs) - 1)
    return fluxes, fluxerrs


# distance modulus fitting function
@tf.function
def distance_modulus(z):
    return 5*tf.math.log(1e6*A0_*(1+z)*(eta0_ - (B0_*((1+z)**4 + B1_*(1+z)**3 + B2_*(1+z)**2 + B3_*(1+z) + B4_)**(-0.125) )))/ln10_ - 5

# comiving distance fitting function
@tf.function
def comoving_distance(z):

	return 1e6*A0_*(eta0_ - (B0_*((1+z)**4 + B1_*(1+z)**3 + B2_*(1+z)**2 + B3_*(1+z) + B4_)**(-0.125) ))

# comoving volume elemebt
@tf.function
def dVdz(z):

    return  comoving_distance(z)**2 * A0_ * (eta0_ + (0.125 * B0_* ((1+z)**4 + B1_*(1+z)**3 + B2_*(1+z)**2 + B3_*(1+z) + B4_)**(-1.125) *  (4*(1+z)**3 + 3*B1_*(1+z)**2 + 2*B2_*(1+z) + B3_) ))


def load_mcmc(n_sets, dir_chains, prefix='baseline'):
    samples = []
    loglike = []
    for i in range(n_sets):
        try:
            s = np.load(dir_chains + 'chain_' + prefix + '_batch{}.npy'.format(i)).astype(np.float32)[-1, :, :]
            l = np.load(dir_chains + 'loglike_' + prefix + '_batch{}.npy'.format(i)).astype(np.float32)
            samples.append(s)
            loglike.append(l)
        except:
            #print('Couldnt load training files '+str(i))
            continue
    samples = np.concatenate(samples, axis=1)
    loglike = np.concatenate(loglike, axis=1)
    return samples, loglike


def load_training_data(root_dir, modelname, n_sets=40, exclude_first=True, prefix='COSMOS20'):
    assert prefix == 'KV' or prefix == 'COSMOS20'

    training_theta = []
    training_absmags = []
    for i in range(n_sets):
        fname = root_dir+'/'+modelname+'/training_data'
        try:
            theta = np.load(root_dir+'/'+modelname+'/training_data/parameters/parameters{}.npy'.format(i)).astype(np.float32)
            absmags = np.load(root_dir+'/'+modelname+'/training_data/photometry/'+prefix+'_photometry{}.npy'.format(i)).astype(np.float32)
            training_theta.append(theta)
            training_absmags.append(absmags)
        except:
            print('Couldnt load training files '+str(i), fname)
            continue
    training_theta = np.concatenate(training_theta, axis=0)
    training_absmags = np.concatenate(training_absmags, axis=0)

    print('modelname: ', modelname)
    if modelname == 'model_HMI':
    # re-parameterization
        training_theta[:, 2] = np.sqrt(training_theta[:, 2]) # dust2 -> sqrt(dust2)
    elif modelname == 'model_Prospector-alpha':
        # re-parameterization
        training_theta[:,-7] = np.sqrt(training_theta[:,-7]) # dust2 -> sqrt(dust2)
        training_theta[:,-4] = np.log(training_theta[:,-4]) # fagn log
        training_theta[:,-3] = np.log(training_theta[:,-3]) # agntau log
    elif modelname == 'model_A':
        training_theta[:, 2] = np.sqrt(training_theta[:, 2]) # dust2 -> sqrt(dust2)
    elif modelname == 'model_B':
        training_theta[:, 2] = np.sqrt(training_theta[:, 2]) # dust2 -> sqrt(dust2)
    elif modelname == 'model_HMII':
        training_theta[:, 2] = np.sqrt(training_theta[:, 2]) # dust2 -> sqrt(dust2)
    else:
        print('ERROR')
        exit(1)

    s = np.all(np.isfinite(training_theta), axis=1)
    s &= np.all(np.isfinite(training_absmags), axis=1)
    s &= training_theta[:, 1] < 0.19 # make sure log10Z spans [-1.98, 0.19] - hardcoded until prior is regenerated
    training_theta = training_theta[s, :]
    training_absmags = training_absmags[s, :]
    print('Kept', np.sum(s), 'out of', s.size)

    if prefix == 'KV':
        filternames = ['omegacam_' + n for n in ['u','g','r','i']] + ['VISTA_' + n for n in ['Z','Y','J','H','Ks']]
    if  prefix == 'COSMOS15':
        filternames = ['ip_cosmos', 'v_cosmos', 'uvista_y_cosmos', 'r_cosmos', 'hsc_y',
           'zpp', 'b_cosmos', 'uvista_h_cosmos', 'wircam_H', 'ia484_cosmos',
           'ia527_cosmos', 'ia624_cosmos', 'ia679_cosmos', 'ia738_cosmos',
           'ia767_cosmos', 'ia427_cosmos', 'ia464_cosmos', 'ia505_cosmos',
           'ia574_cosmos', 'ia709_cosmos', 'ia827_cosmos', 'uvista_j_cosmos',
           'uvista_ks_cosmos', 'wircam_Ks', 'NB711.SuprimeCam',
           'NB816.SuprimeCam']
    if  prefix == 'COSMOS20':
        filternames = [
           'galex_NUV', 'u_megaprime_sagem',
           'hsc_g', 'hsc_r', 'hsc_i', 'hsc_z', 'hsc_y',
           'uvista_y_cosmos', 'uvista_j_cosmos', 'uvista_h_cosmos', 'uvista_ks_cosmos',
           'ia427_cosmos', 'ia464_cosmos', 'ia484_cosmos', 'ia505_cosmos', 'ia527_cosmos',
           'ia574_cosmos', 'ia624_cosmos', 'ia679_cosmos', 'ia709_cosmos', 'ia738_cosmos',
           'ia767_cosmos', 'ia827_cosmos',
           'NB711.SuprimeCam', 'NB816.SuprimeCam',
           'b_cosmos', 'v_cosmos', 'r_cosmos', 'ip_cosmos', 'zpp',
           'irac1_cosmos', 'irac2_cosmos'
        ]

    if exclude_first:
        training_theta = training_theta[:, 1:]

    return training_theta, training_absmags, filternames



def make_mock(
fluxes, redshifts, filter_names, n_obj=15000,
    band_snr_cuts=[('BLAG', 10)]
    ):

    num_bands = len(filter_names)
    magzeropoints = np.ones((num_bands, ))

    #flux_sigmas =
    #fluxes =
    cut = (redshifts < 2.0) * (redshifts > 1e-3)
    for bandname, snrcut in band_snr_cuts:
        if bandname not in filter_names:
            print('Error:', bandname, 'was not found')
        i = bandname
        cut &= np.abs(fluxes[:, i]/flux_sigmas[:, i]) > snrcut

    zspec = redshifts
    zspecsource = np.repeat('', n_obj)

    os.makedirs(directory+'/mock'+suffix)

    np.save(directory+'/mock'+suffix+'/magzeropoints.npy', magzeropoints)
    np.save(directory+'/mock'+suffix+'/filternames.npy', filter_names)
    np.save(directory+'/mock'+suffix+'/fluxes.npy', fluxes)
    np.save(directory+'/mock'+suffix+'/flux_errors.npy', flux_sigmas)
    np.save(directory+'/mock'+suffix+'/specz.npy', zspec)
    np.save(directory+'/mock'+suffix+'/photoz.npy', zspec)
    np.save(directory+'/mock'+suffix+'/specz_source.npy', zspecsource)

    # cut out dodgy values

    return magzeropoints, filter_names, fluxes, flux_sigmas, zspec, specsource, photoz


def load_cosmos_data(directory, zspeconly=True):
    #magzeropoints = np.load(directory+'/magzeropoints1.npy').astype(np.float32)
    filter_names = np.load(directory+'/filternames.npy')
    fluxes = np.load(directory+'/fluxes.npy').astype(np.float32)
    flux_sigmas = np.load(directory+'/flux_errors.npy').astype(np.float32)
    zspec = np.load(directory+'/specz.npy').astype(np.float32)
    try:
        from astropy.table import Table
        photoz = Table.read(directory+'/photoz.fits')
        specsource = np.load(directory+'/specz_source.npy')
    except:
        photoz = np.repeat(-1, zspec.size)
        specsource = np.repeat('', zspec.size)

    # cut out dodgy values
    if zspeconly:
        cut = (zspec < 2.0) * (zspec > 1e-3)# * (specsource != 'CDFS') * (specsource != 'VVDS')
        print('Selected', np.sum(cut), 'objects out of', cut.size)
        fluxes = fluxes[cut, :]
        flux_sigmas = flux_sigmas[cut, :]
        zspec = zspec[cut]
        specsource = specsource[cut]
        photoz = photoz[cut]
    else:
        print(zspec.size, 'objects')

    #* 1e29 # erg/cm2/s/Hz to uJy
    #/ 3.631 # uJy to nanomaggy
    # 1 nanomaggy = 3.631×10-6 Jy = 3.631 uJy
    # units: from erg/cm2/s/Hz to nanomaggies
    fluxes *= 1e29 / 3.631
    flux_sigmas *= 1e29 / 3.631

    used_filters = np.array([
       #'galex_NUV',
       'u_megaprime_sagem',
       'hsc_g', 'hsc_r', 'hsc_i', 'hsc_z', 'hsc_y',
       'uvista_y_cosmos', 'uvista_j_cosmos', 'uvista_h_cosmos', 'uvista_ks_cosmos',
       'ia427_cosmos', 'ia464_cosmos', 'ia484_cosmos', 'ia505_cosmos', 'ia527_cosmos',
       'ia574_cosmos', 'ia624_cosmos', 'ia679_cosmos', 'ia709_cosmos', 'ia738_cosmos',
       'ia767_cosmos', 'ia827_cosmos',
       'NB711.SuprimeCam', 'NB816.SuprimeCam',
       #'b_cosmos', 'v_cosmos', 'r_cosmos', 'ip_cosmos', 'zpp',
       'irac1_cosmos', 'irac2_cosmos'
    ])

    if 'Classic' in directory:
        zpa = 10**(0.4*np.load(directory+'/magzeropoints2.npy'))
        zpb = 10**(0.4*np.load(directory+'/magzeropoints4.npy'))
    if 'Farmer' in directory:
        zpa = 10**(0.4*np.load(directory+'/magzeropoints1.npy'))
        zpb = 10**(0.4*np.load(directory+'/magzeropoints3.npy'))

    if True:
        ind = np.in1d(filter_names, used_filters)
        print('Using', np.sum(ind), 'filters out of', filter_names.size)
        filter_names = filter_names[ind]
        fluxes = fluxes[:, ind]
        flux_sigmas = flux_sigmas[:, ind]
        zpa = zpa[ind]
        zpb = zpb[ind]

    return filter_names, fluxes, flux_sigmas, zspec, photoz, specsource, zpa, zpb



def assert_no_nan_or_inf(x):
    c1 = tf.math.count_nonzero(tf.math.is_nan(x)).numpy()
    c2 = tf.math.count_nonzero(tf.math.is_inf(x)).numpy()
    assert c1 == 0
    assert c2 == 0

def load_initial_hyperparameters(dir_init_with_prefix):
    hyper_parameters = np.load(dir_init_with_prefix + 'hyper_parameters_init.npy')
    return hyper_parameters


from scipy.stats import median_abs_deviation
from scipy.stats import binned_statistic

def sig68_fn(diff):
    if diff.size <= 10:
        return 0
    else:
        p = np.percentile(diff, np.array([16, 84]))
        return (p[1] - p[0]) / 2

def signmad_fn(diff):
    return 1.48 * np.median(np.abs(diff))

def out_fn(diff):
    return 100 * np.sum(np.abs(diff) > 0.15) / diff.size

def metrics(z_phot, z_spec):
    diff = (z_phot - z_spec) / (1 + z_spec)
    ind = z_phot > 0
    if np.sum(~ind) > 0 :
        print(np.sum(~ind), 'obj (out of', ind.size, ') with z_phot=0 (excluded for those statistics)')
    diff = diff[ind]
    z_spec = z_spec[ind]
    print('Mean: %.3f' % np.mean(diff), end="  ")
    print('Median: %.3f' % np.median(diff), end="  ")
    print('Mean stddev: %.3f' % np.std(diff), end="  ")
    print('Median stddev: %.3f' % median_abs_deviation(diff), end="  ")
    print('Sigma 68: %.3f' % sig68_fn(diff), end="  ")
    print('Outliers: %.3f' % out_fn(diff))





def load_lines_model(dir_spsmodels, modelname, filternames):

    restore_filename = dir_spsmodels + '/' + modelname + '/trained_models/speculator-emlinesabsmags'
    speculator_emlines = Speculator(restore=True, restore_filename=restore_filename)

    line_lambdas = speculator_emlines.wavelengths
    #line_lambdas_selected = np.array([4862.71,  6564.6 , 3729.86, 5008.24, 4341.69])#, 9533.2, 9071.1])
    line_lambdas_selected2 = np.array([
        1215.67,  2471.09,  3109.98,  3727.1 ,  3869.86,
        3971.2 ,  4069.75,  4341.69,  4725.47,  4960.3 ,  5008.24,
        5193.27,  5756.19,  6564.6 ,  6718.29,  7067.14,  9071.1 ,
        9533.2 , 10052.6 , 10832.06, 12821.58, 18756.4
        ])
    line_names2 = np.array(['H I (Ly-alpha)', '[O II] 2471', '[Ar III] 3110', '[O II] 3726',
       '[Ne III] 3870', 'H-5 3970', '[S II] 4070', 'H-gamma 4340',
       '[Ne IV] 4720', '[O III] 4960', '[O III] 5007', '[Ar III] 5193',
       '[N II] 5756', 'H-alpha 6563', '[S II] 6717', 'He I 7065',
       '[S III] 9071', '[S III] 9533', 'H I (Pa-delta)', 'He I 10829',
       'H I (Pa-beta)', 'H I (Pa-alpha)'])

    data = np.genfromtxt('/home/bl/Dropbox/repos/steppz/apjaa6c66t4_ascii.txt', delimiter='\t', dtype=str, skip_header=True)
    line_lambdas_selected = data[:, 0].astype(float)
    line_names = data[:, 1]

    if True:
        line_lambdas_selected = np.array([17366.885 ,  2326.11  ,  2321.664 ,  6302.046 ,  4069.75  ,
        1215.6701,  2669.951 , 30392.02  ,  7753.19  , 37405.76  ,
       32969.8   , 18179.2   ,  9017.8   ,  2661.146 ,  6313.81  ,
        9232.2   ,  3722.75  ,  2803.53  , 19450.89  ,  9548.8   ,
        7067.138 ,  6549.86  ,  6732.673 , 10052.6   ,  1908.73  ,
        6679.995 ,  2796.352 ,  6718.294 , 21661.178 , 40522.79  ,
        7137.77  , 10833.306 ,  1906.68  , 10941.17  ,  4472.735 ,
        4364.435 ,  6585.27  , 12821.578 , 26258.71  ,  9071.1   ,
        3798.987 , 10832.057 ,  3889.75  ,  3836.485 ,  3968.59  ,
        5877.249 ,  3890.166 ,  9533.2   ,  3971.198 , 18756.4   ,
        3727.1   ,  4102.892 ,  3729.86  ,  3869.86  ,  4341.692 ,
        4960.295 ,  4862.71  ,  6564.6   ,  5008.24  ])
        line_names = np.array([
            'H I (Br-6)', 'C II] 2326', '[O III] 2321', '[O I] 6302', '[S II] 4070',
            'H I (Ly-alpha)', '[Al II] 2670', 'H I (Pf-5)', '[Ar III] 7753', 'H I (Pf-gamma)',
             'H I (Pf-delta)', 'H I (Br-5)', 'H I (Pa-7)', '[Al II] 2660', '[S III] 6314', 'H I (Pa-6)',
             '[S III] 3723', 'Mg II 2800', 'H I (Br-delta)', 'H I (Pa-5)', 'He I 7065', '[N II] 6549',
             '[S II] 6732', 'H I (Pa-delta)', 'C III]', 'He I 6680', 'Mg II 2800', '[S II] 6717',
             'H I (Br-gamma)', 'H I (Br-alpha)', '[Ar III] 7138', 'He I 10833', '[C III]',
             'H I (Pa-gamma)', 'He I 4472', '[O III] 4364', '[N II] 6585', 'H I (Pa-beta)',
             'H I (Br-beta)', '[S III] 9071', 'H-8 3798', 'He I 10829', 'He I 3889', 'H-7 3835',
             '[Ne III] 3968', 'He I 5877', 'H-6 3889', '[S III] 9533', 'H-5 3970', 'H I (Pa-alpha)',
              '[O II] 3726', 'H-delta 4102', '[O II] 3729', '[Ne III] 3870', 'H-gamma 4340', '[O III] 4960',
              'H-beta 4861', 'H-alpha 6563', '[O III] 5007'])
        #ind = ~np.in1d(line_lambdas_selected2, line_lambdas_selected)
        #line_lambdas_selected2 = line_lambdas_selected2[ind]
        #line_names2 = line_names2[ind]
        #line_lambdas_selected = np.concatenate([line_lambdas_selected, line_lambdas_selected2])
        #line_names = np.concatenate([line_names, line_names2])

    ind = line_lambdas_selected > 1e3
    ind &= line_lambdas_selected < 1e4
    line_names = line_names[ind]
    line_lambdas_selected = line_lambdas_selected[ind]

    n_lines = line_lambdas_selected.size
    line_idx_selected = []
    for l in line_lambdas_selected:
        diff = (l - line_lambdas)**2
        loc = np.argmin(diff)
        if diff[loc] > 1.0:
            print('Not loading line', l, 'because nearest is', line_lambdas[l])
        line_idx_selected += [loc]
    line_idx_selected = np.array(line_idx_selected)
    assert np.max(np.abs(line_lambdas[line_idx_selected] - line_lambdas_selected)) < 10


    bandCoefs = tf.convert_to_tensor(
            np.concatenate([np.load('/home/bl/Dropbox/repos/steppz/filters_gengmm/line_gengmmcoeffs_'+filter+'.npy').astype(np.float32)[None, :]
        for filter in filternames]))
    n = bandCoefs.shape[1] // 4
     # n_bands, num_GMM
    gengnn_amps = bandCoefs[:, 0:n]
    gengnn_sigs = bandCoefs[:, n:2*n]
    gengnn_betas = bandCoefs[:, 2*n:3*n]
    gengnn_locs = bandCoefs[:, 3*n:4*n]
    lines_absfluxes_gengmm = tfd.GeneralizedNormal(
        loc=gengnn_locs[None, None, :, :], scale=gengnn_sigs[None, None, :, :], power=gengnn_betas[None, None, :, :]
    ) # 1, 1, n_bands, num_GMM

    return n_lines, speculator_emlines, line_lambdas_selected, line_idx_selected, gengnn_amps, lines_absfluxes_gengmm, line_names



def process_prefix(prefix):

    prefix_ = prefix.split('_')
    zpidx = prefix_[0] #'a'
    floorerr = float(prefix_[1]) #0.001
    lineserr = float(prefix_[2]) #1.0
    SFSprior = prefix_[3] #SFSprior: mizuki or leja or None
    FMRprior = prefix_[4] #FMRprior: curti or None
    scatter_logzs = float(prefix_[5])

    if prefix_[6] == 'dVdz':
        from priors import redshift_volume_prior
        redshift_prior = redshift_volume_prior
    else:
        redshift_prior = None
    if prefix_[-1] == 'atz':
        at_specz = True
    else:
        at_specz = False

    return zpidx, floorerr, lineserr, SFSprior, FMRprior, scatter_logzs, redshift_prior, at_specz
