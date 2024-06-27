#!/usr/bin/bash
# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com

class OneEuroFilter:
    class LowPassFilter:
        def __init__(self, alpha, initval=0.0):
            self.y = self.s = initval
            self.a = alpha
            self.initialized = False

        def set_alpha(self, alpha):
            if not (0.0 < alpha <= 1.0):
                raise Exception("alpha should be in (0.0, 1.0]")
            self.a = alpha

        def filter(self, value):
            if self.initialized:
                result = self.a * value + (1.0 - self.a) * self.s
            else:
                result = value
                self.initialized = True

            self.y = value
            self.s = result
            return result

        def filter_with_alpha(self, value, alpha):
            self.set_alpha(alpha)
            return self.filter(value)

        def has_last_raw_value(self):
            return self.initialized

        def last_raw_value(self):
            return self.y

    def __init__(self, freq, min_cutoff=1.0, beta_=0.007, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta_
        self.d_cutoff = d_cutoff
        self.x = self.LowPassFilter(self.alpha(min_cutoff))
        self.dx = self.LowPassFilter(self.alpha(d_cutoff))
        self.last_time = -1

    def alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (6.2831855 * cutoff)
        return 1.0 / (1.0 + tau / te)

    def set_frequency(self, f):
        if f <= 0.0:
            raise Exception("freq should be >0")
        self.freq = f

    def set_min_cutoff(self, mc):
        if mc <= 0.0:
            raise Exception("minCutoff should be >0")
        self.min_cutoff = mc

    def set_beta(self, b):
        self.beta = b

    def set_derivative_cutoff(self, dc):
        if dc <= 0.0:
            raise Exception("dCutoff should be >0")
        self.d_cutoff = dc

    def filter(self, value, timestamp=-1):
        if self.last_time != -1 and timestamp != -1:
            self.freq = 1000.0 / (timestamp - self.last_time)

        self.last_time = timestamp
        d_value = (value - self.x.last_raw_value()) * self.freq if self.x.has_last_raw_value() else 0.0
        ed_value = self.dx.filter_with_alpha(d_value, self.alpha(self.d_cutoff))
        cutoff = self.min_cutoff + self.beta * abs(ed_value)
        return self.x.filter_with_alpha(value, self.alpha(cutoff))

    def filter_values(self, values, timestamp=-1):
        if len(values) == 1:
            return self.filter(values[0], timestamp)
        return [self.filter(val, timestamp) for val in values]

