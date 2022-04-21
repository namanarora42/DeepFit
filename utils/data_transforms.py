from scipy.signal import resample_poly
from fractions import Fraction


class Unit(object):
    def __call__(self, sample):
        return sample


class Resample(object):
    def __init__(self, target_length):
        self.target_length = target_length

    def __call__(self, sample):
        cur_len = sample.shape[1]
        frac = Fraction((self.target_length + 1) / cur_len).limit_denominator(100)
        sample = resample_poly(sample, frac.numerator, frac.denominator, axis=1)
        return sample[:, :self.target_length]
