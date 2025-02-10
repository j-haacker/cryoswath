import numpy as np

from cryoswath.l1b import noise_val

def test_noise_val():
    n = 30  # noise_val considers slices with 30 sample thickness
    test_vec_len = 256
    # model thermal noise approximately using normal distribution
    np.random.seed(0)
    test_vec__pure_noise = np.random.normal(size=(test_vec_len,))
    test_vec__linear_trend = test_vec__pure_noise + np.linspace(0, 1, test_vec_len)
    # distance between noise and signal are +- 10 standard deviabtions
    test_vec__step_start_3rd_slice = test_vec__pure_noise + 10*(np.arange(test_vec_len) >= 2*n)
    assert noise_val(test_vec__pure_noise) == np.mean(test_vec__pure_noise)
    assert noise_val(test_vec__linear_trend) == np.mean(test_vec__linear_trend)
    assert noise_val(test_vec__step_start_3rd_slice) == np.mean(test_vec__step_start_3rd_slice[:2*n])
    # edge cases
    test_vec__step_start_2nd_slice = test_vec__pure_noise + 10*(np.arange(test_vec_len) >= n)
    assert noise_val(test_vec__step_start_2nd_slice) == np.mean(test_vec__step_start_2nd_slice[:n])
    test_vec__step_mid_2nd_slice = test_vec__pure_noise + 10*(np.arange(test_vec_len) >= 1.5*n)
    assert noise_val(test_vec__step_mid_2nd_slice) <= np.mean(test_vec__step_mid_2nd_slice[:2*n])
    test_vec__exp_increase = test_vec__pure_noise + 10**(np.linspace(0, 1, test_vec_len))
    assert noise_val(test_vec__exp_increase) < np.mean(test_vec__exp_increase)

test_noise_val()