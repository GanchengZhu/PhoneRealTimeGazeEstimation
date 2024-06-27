#!/usr/bin/bash
# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com


class HeuristicFilterManager:
    def __init__(self, look_ahead):
        self.raw_x = []
        self.raw_y = []
        self.dummy_x = 0.0
        self.dummy_y = 0.0
        self.look_ahead = look_ahead

    def filter_values(self, timestamp, x, y):
        self.raw_x = self.do_filter(True, x)
        self.raw_y = self.do_filter(False, y)

    def do_filter(self, is_x, element):
        raw = self.raw_x if is_x else self.raw_y
        raw.append(element)
        if len(raw) == self.look_ahead * 2 + 1:
            for next_val in range(1, self.look_ahead + 1):
                condition_one = raw[self.look_ahead - next_val] < raw[self.look_ahead] and raw[self.look_ahead] > raw[self.look_ahead + next_val]
                condition_two = raw[self.look_ahead - next_val] > raw[self.look_ahead] and raw[self.look_ahead] < raw[self.look_ahead + next_val]
                if condition_one or condition_two:
                    prev_dist = abs(raw[self.look_ahead - next_val] - raw[self.look_ahead])
                    next_dist = abs(raw[self.look_ahead + next_val] - raw[self.look_ahead])
                    if prev_dist < next_dist:
                        raw[self.look_ahead] = raw[self.look_ahead - next_val]
                    else:
                        raw[self.look_ahead] = raw[self.look_ahead + next_val]
            if is_x:
                self.dummy_x = raw[self.look_ahead]
            else:
                self.dummy_y = raw[self.look_ahead]
            raw.pop(0)
        return raw

    def get_filtered_values(self):
        if len(self.raw_x) == self.look_ahead * 2:
            return [self.dummy_x, self.dummy_y]
        else:
            return None
