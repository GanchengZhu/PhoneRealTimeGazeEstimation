class OneEuroFilter:
    def __init__(self, freq, minCutoff=1.0, beta_=0.007, dCutoff=1.0):
        self.freq = freq
        self.minCutoff = minCutoff
        self.beta_ = beta_
        self.dCutoff = dCutoff
        self.x = self.LowPassFilter(self.alpha(minCutoff))
        self.dx = self.LowPassFilter(self.alpha(dCutoff))
        self.lastTime = -1

    def alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (6.2831855 * cutoff)
        return 1.0 / (1.0 + tau / te)

    def setFrequency(self, f):
        if f <= 0.0:
            raise Exception("freq should be >0")
        self.freq = f

    def setMinCutoff(self, mc):
        if mc <= 0.0:
            raise Exception("minCutoff should be >0")
        self.minCutoff = mc

    def setBeta(self, b):
        self.beta_ = b

    def setDerivateCutoff(self, dc):
        if dc <= 0.0:
            raise Exception("dCutoff should be >0")
        self.dCutoff = dc

    def init(self, freq, mincutoff, beta_, dcutoff):
        self.setFrequency(freq)
        self.setMinCutoff(mincutoff)
        self.setBeta(beta_)
        self.setDerivateCutoff(dcutoff)
        self.x = self.LowPassFilter(self.alpha(mincutoff))
        self.dx = self.LowPassFilter(self.alpha(dcutoff))
        self.lastTime = -1

    def filter(self, value, timestamp=-1):
        if self.lastTime != -1 and timestamp != -1:
            self.freq = 1000.0 / (timestamp - self.lastTime)
        self.lastTime = timestamp
        dvalue = (value - self.x.lastRawValue()) * self.freq if self.x.hasLastRawValue() else 0.0
        edvalue = self.dx.filterWithAlpha(dvalue, self.alpha(self.dCutoff))
        cutoff = self.minCutoff + self.beta_ * abs(edvalue)
        return self.x.filterWithAlpha(value, self.alpha(cutoff))

    class LowPassFilter:
        def __init__(self, alpha, initval=0.0):
            self.y = self.s = initval
            self.a = alpha
            self.initialized = False

        def setAlpha(self, alpha):
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

        def filterWithAlpha(self, value, alpha):
            self.setAlpha(alpha)
            return self.filter(value)

        def hasLastRawValue(self):
            return self.initialized

        def lastRawValue(self):
            return self.y


class OneEuroFilterManager:
    def __init__(self, count=1, freq=30.0, minCutOff=1.0, beta=0.007, dCutOff=1.0):
        self.initFilter(count, freq, minCutOff, beta, dCutOff)

    def initFilter(self, count, freq, minCutOff, beta, dCutOff):
        if freq <= 0.0:
            freq = 30.0

        if minCutOff <= 0.0:
            minCutOff = 1.0

        if dCutOff <= 0.0:
            dCutOff = 1.0

        try:
            self.filters = [OneEuroFilter(freq, minCutOff, beta, dCutOff) for _ in range(count)]
            self.filteredValues = [0.0] * count
            self.count = count
            self.freq = freq
            self.minCutOff = minCutOff
            self.beta = beta
            self.dCutOff = dCutOff
        except Exception as e:
            print("Filter init fail:", e)

    def isInvalidValue(self, val):
        return val == float('inf') or val == float('nan')

    def filterMultipleValue(self, timestamp, *vals):
        if vals is not None and len(vals) == len(self.filters):
            try:
                for idx, val in enumerate(vals):
                    filteredVal = self.filters[idx].filter(val, timestamp)
                    if self.isInvalidValue(filteredVal):
                        self.initFilter(self.count, self.freq, self.minCutOff, self.beta, self.dCutOff)
                        return False
                    self.filteredValues[idx] = filteredVal
                return True
            except Exception as e:
                return False
        else:
            return False

    def filterValues(self, timestamp, x, y):
        return self.filterMultipleValue(timestamp, x, y)

    def getFilteredValues(self):
        return self.filteredValues
