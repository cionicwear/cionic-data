# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import tools
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
import functools
import operator
import struct
import time
import scipy.stats as stats
from scipy.stats import ttest_ind
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
from scipy.spatial.distance import cdist
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as pltk
import math
import logging

def GaitCycleAxis():
    return {
        'lim'    : [0,1],
        'format' : pltk.PercentFormatter(xmax=1),
        'label'  : '% of gait cycle'
    }

def CycleAxis():
    return {
        'lim'    : [0,1],
        'format' : pltk.PercentFormatter(xmax=1),
        'label'  : '% of cycle'
    }

def NumberAxis(ymin, ymax, label):
    return {
        'lim'    : [ymin,ymax],
        'format' : pltk.FormatStrFormatter("%d"),
        'label'  : label,
    }

def JointAngleAxis(ymin, ymax):
    return {
        'lim'    : [ymin,ymax],
        'format' : pltk.FormatStrFormatter("%d°"),
        'label'  : 'Joint Angle',
    }

def EMGAxis(ymin, ymax):
    return {
        'lim'    : [ymin,ymax],
        'format' : pltk.PercentFormatter(xmax=1),
        'label'  : '% of mean',
    }

def PressureAxis(ymin, ymax):
    return {
        'lim'    : [ymin,ymax],
        'format' : pltk.PercentFormatter(xmax=1),
        'label'  : '% of max',
    }

# Presentation versions of the Axis that remove all the labels

def GaitCycleSimpleAxis():
    return {
        'lim'    : [0,1],
        'format' : pltk.FuncFormatter(lambda x, pos: f"{int(x*100)}"),
        'label'  : ''
    }

def EMGSimpleAxis(ymin, ymax):
    return {
        'lim'    : [ymin,ymax],
        'format' : pltk.FuncFormatter(lambda x, pos: f"{int(x*100)}"),
        'label'  : '',
    }

def JointAngleSimpleAxis(ymin, ymax):
    return {
        'lim'    : [ymin,ymax],
        'format' : pltk.FormatStrFormatter("%d"),
        'label'  : '',
    }

def cdist_dtw(x, y):
    if np.ndim(x) == 1:
        x = x.reshape(-1, 1)
    if np.ndim(y) == 1:
        y = y.reshape(-1, 1)

    matrix_0 = np.zeros((len(x) + 1, len(y) + 1))
    matrix_0[0, 1:] = np.inf
    matrix_0[1:, 0] = np.inf
    matrix_1 = matrix_0[1:, 1:]
    matrix_0[1:, 1:] = cdist(x, y, 'euclidean')
    for i in range(len(x)):
        for j in range(len(y)):
            min_list = [matrix_0[i, j]]
            for k in range(1, 2):
                min_list += [matrix_0[min(i + k, len(x)), j],
                                 matrix_0[i, min(j + k, len(y))]]
            matrix_1[i, j] += min(min_list)

    return matrix_1[-1, -1]

def costri(a, b, C):
    """ given sides a & b and angle C
        calculate angles A & B and side c
        and return complete triangle definition
    """
    c = math.sqrt(a**2 + b**2 - 2*a*b*math.cos(C))
    return {
        'a' : a,
        'b' : b,
        'c' : c,
        'A' : np.arccos((c**2 + b**2 - a**2) / (2*c*b)),
        'B' : np.arccos((c**2 + a**2 - b**2) / (2*c*a)),
        'C' : C
    }

def get_walking_intervals(
        kinematic_time_series,
        component="x",
        n_start_remove=2,
        n_stop_remove=1,
    ):
    peak_kwargs = {"distance": 50, "prominence": 20}
    peaks, _ = find_peaks(kinematic_time_series[component], **peak_kwargs)
    splits_timestamps = kinematic_time_series["elapsed_s"][peaks]
    if splits_timestamps.shape[0] < 2:
        return [()]
    median_time_interval = np.median(np.diff(splits_timestamps))

    all_grouped_splits = []
    this_group_splits = []
    for i in range(splits_timestamps.shape[0] - 1):
        this_split, next_split = splits_timestamps[i], splits_timestamps[i+1]

        if next_split - this_split < 2 * median_time_interval:
            this_group_splits += [this_split, next_split]
        else:
            all_grouped_splits.append(sorted(list(set(this_group_splits))))
            this_group_splits = []
    all_grouped_splits.append(sorted(list(set(this_group_splits))))

    walking_intervals = []
    for grouped_splits in all_grouped_splits:
        if len(grouped_splits) + 1 > n_start_remove:
            grouped_splits = grouped_splits[n_start_remove:]
        if len(grouped_splits) + 1 > n_stop_remove:
            grouped_splits = grouped_splits[:-n_stop_remove] if n_stop_remove > 0 else grouped_splits
        if len(grouped_splits) < 2:
            continue
        walking_intervals.append((grouped_splits[0], grouped_splits[-1]))

    return walking_intervals

class Kinematics:

    plot_width = 8
    plot_height = 5

    def __init__(self, config):
        # self.groups[group][position][stream]
        self.groups = defaultdict(lambda: defaultdict(lambda: {}))
        # self.calibrations[group][position][stream]
        self.calibrations = defaultdict(lambda: defaultdict(lambda: {}))
        # self.calibrations[group][position][stream]
        self.channels = defaultdict(lambda: defaultdict(lambda: {}))
        # self.regs[group][position][stream]
        self.regs = defaultdict(lambda: defaultdict(lambda: {}))
        # self.splits[group][name]
        self.splits = defaultdict(lambda: {})
        self.time_ranges = defaultdict(lambda: {})
        self.triggers = defaultdict(lambda: {})
        # (group, name) : (position_a, position_b, axis)
        self.config = config

        self.pp = None
        self.study = None
        self.coll_num = None
        self.segments = None

    def save_open(self, outpath):
        self.pp = PdfPages(outpath)

    def save_text(self, txt, width=plot_width, height=plot_height):
        fig = plt.figure()
        fig.set_size_inches(width, height, forward = False)
        # fig.clf()
        fig.text(0.5,0.5,txt, transform=fig.transFigure, size=24, ha="center")
        self.save_fig(fig)

    def save_fig(self, fig):
        if self.pp:
            self.pp.savefig(fig)

    def save_close(self):
        self.pp.close()
        self.pp = None

    def load_array(self, group, position, stream, array, regs, hz=None, time_range=None):

        if hz is not None:
            ts = 0
            inc = (1.0/hz)
            for a in array: 
                a['elapsed_s'] = ts
                ts += inc

        if time_range is not None:
            array = array[(array['elapsed_s'] >= time_range[0]) & (array['elapsed_s'] < time_range[1])]

        if stream in self.groups[group][position]:
            self.groups[group][position][stream] = np.concatenate((self.groups[group][position][stream], array))
        else:
            self.groups[group][position][stream] = array
            self.regs[group][position][stream] = regs

    def load_csv_stream(self, group, position, stream, array, label_range=None, time_range=None):
        print(f"loading {position} {stream} to {group}")
        if label_range is not None:
            array = array[(array['elapsed_s'] >= label_range[0]) & (array['elapsed_s'] < label_range[1])]

        if time_range is not None:
            array = array[(array['elapsed_s'] >= time_range[0]) & (array['elapsed_s'] < time_range[1])]

        if stream in self.groups[group][position]:
            self.groups[group][position][stream] = np.concatenate((self.groups[group][position][stream], array))
        else:
            self.groups[group][position][stream] = array
        

    def check(self, group, position=None, stream=None, component=None):
        if group not in self.groups:
            print(f"no group {group}")
            return False

        if position and position not in self.groups[group]:
            print(f"no position {position} in {group}")
            return False

        if stream and stream not in self.groups[group][position]:
            print(f"no stream {stream} for {position} in {group}")
            return False

        if component and component not in self.groups[group][position][stream].dtype.names:
            print(f"no component {component} in {stream} for {position} in {group}")
            return False

        return True

    ######################
    # split calculations #
    ######################

    def calculate_segment_splits(self, group, label, name, streams):
        splits = []

        for index, seg in streams.iterrows():
            if seg['label'] == label:
                splits.append((seg['start_s'], seg['end_s'], 1.0))

        self.splits[group][name] = {
            'splits' : splits,
            'skips' : []
        }

    def calculate_splits(self, pconf, skip_splits_outside_valid_range=True):
        def check_if_timestamp_in_valid_range(timestamp, time_ranges):
            for interval_start, interval_stop in time_ranges:
                if timestamp >= interval_start and timestamp <= interval_stop:
                    return True
            return False

        # pconf contains a dictionary of splitters keyed on name
        for name, splitter in pconf.items():

          # a splitter contains a dictionary of split configs keyed on group
            for group, conf in splitter.items():

                (position, stream, component) = conf['split']
                skips = conf.get('skips', [0])

                # confirm this component exists
                if not self.check(group, position, stream):
                    continue

                # load the stream and split
                s = self.groups[group][position][stream][component]
                elapsed = self.groups[group][position][stream]['elapsed_s']
                splits = conf['func'](s, conf['config'], elapsed)

                # add skips that are outside viable time ranges
                if skip_splits_outside_valid_range:
                    for idx, split in enumerate(splits):
                        ts, _ = split
                        in_valid_range = check_if_timestamp_in_valid_range(ts, self.time_ranges[group])
                        if not in_valid_range  and idx not in skips:
                            skips.append(idx)

                # create a stream that can be graphed
                splits_stream = []
                if splits:
                    s = 0
                    (ts, height) = splits[s]

                    for (i, t) in enumerate(elapsed):
                        if t == ts:
                            if s in skips:
                                splits_stream.append((t, -1.0))
                            else:
                                splits_stream.append((t, height))
                            s += 1
                            if s < len(splits):
                                (ts, height) = splits[s]
                        else:
                            if s in skips:
                                splits_stream.append((t, -1.0))
                            else:
                                splits_stream.append((t, 0))

                self.splits[group][name] = {
                    'splits' : splits,
                    'skips'   : skips,
                    'streams' : np.array(splits_stream, dtype={'names':('elapsed_s', 'peak'),'formats':('f8', 'f8')})
                }

    def set_contras(self, a, b, splits):
        for split in splits:
            cl = f"contra_{split}"
            if split in self.splits[a]:
                self.splits[b][cl] = self.splits[a][split]
            if split in self.splits[b]:
                self.splits[a][cl] = self.splits[b][split]


    def split_times(self, side, a, b):
        """ given two splits calculate the timings between
            accounting for skips and missing data
        """
        start = self.splits[side][a]
        end = self.splits[side][b]
        times = []
        for i,s in enumerate(start['splits']):

            if i in start['skips']:
                continue
            if i < len(start['splits'])-1:
                next_ts = start['splits'][i+1][0]
            else:
                next_ts = start['splits'][i][0] + 100 # todo: use end of recording

            for j, e in enumerate(end['splits']):
                if j in end['skips']:
                    continue
                if e[0] > s[0] and e[0] <= next_ts:
                    times.append(e[0]-s[0])
                    break
        return times

    def export_split(self, split_name, group, position, stream, component, target_length):
        """ output emg splits into csv format where each row is an individual stream of target_length points
            <stream> <start> <duration> <0>...<target_length>
        """
        splits = []

        if not self.splits.get(group, []):
            print(f"no splits for {group}")
            return splits

        if not self.check(group, position, stream, component):
            return splits

        vals = self.groups[group][position][stream]
        a = 0
        split = self.splits[group].get(split_name)

        if not split:
            print(f"split {split_name} not in group {group}")
            return splits

        for i, strike in enumerate(split['splits']):
            (ts,height) = strike

            if i == len(split['splits'])-1:
                break

            # get the next strike to calculate duration
            duration = split['splits'][i+1][0] - ts

            arr = []
            while a < len(vals['elapsed_s']) and vals['elapsed_s'][a] < ts:
                arr.append(vals[component][a])
                a += 1

            if i not in split['skips'] and len(arr) > 0:
                (time_interp, step_interp) = self.interpolate_split(arr, target_length)
                splits.append(np.insert(step_interp, 0, [ts, duration]))

        return splits


    #############
    # splitting #
    #############

    def interpolate_split(self, split, target_length):
        # normalize step and time
        len_split = len(split)
        time_orig = np.linspace(0,1,len_split)
        time_interp = np.linspace(0,1,target_length)
        step_interp = np.interp(time_interp, time_orig, split)
        return (time_interp, step_interp)

    def split_stream(self, split_name, group, position, stream, component):
        """
        create an array of equal sized angle arrays
        """
        vals = self.groups[group][position][stream]
        start = 0
        splits = []
        split = self.splits[group][split_name]
        for i, strike in enumerate(split['splits']):
            if len(strike) == 3:
                # strike with start and end
                (start, end, height) = strike
            else:
                # strike with only end
                (end, height) = strike

            sub = vals[(vals["elapsed_s"] >= start) & (vals["elapsed_s"] < end)]
            start = end

            if i not in split['skips']:
                splits.append({
                    component : sub[component],
                    'elapsed_s' : sub['elapsed_s']
                })
        return splits

    def calculate_split(self, split_name, group, position, stream, component, target_length=2500, rms_emg_after_seg=None):
        if not isinstance(rms_emg_after_seg, dict):
            rms_emg_after_seg = {'action': False}

        splits = self.split_stream(split_name, group, position, stream, component)

        steps = np.empty((0,target_length))
        steps_ind = []
        for split in splits:
            if len(split[component]) == 0:
                continue

            if component == 'emg' and rms_emg_after_seg.get('action', False):
                rmsw = rms_emg_after_seg.get('rms_window', 301)
                norm = rms_emg_after_seg.get('normalize')
                split[component] = tools.moving_avg_rms(split[component], window_size=rmsw)
                if norm:
                    split[component] = split[component] / norm(split[component])

            (time_interp, step_interp) = self.interpolate_split(split[component], target_length)
            steps = np.append(steps, [step_interp], axis = 0)
            steps_ind.append((time_interp, step_interp))

        return {
            'avg' : steps.mean(axis=0),
            'std' : steps.std(axis=0),
            'ind' : steps_ind
        }

    ####################
    # kinematic scores #
    ####################
       
    def normativescore(self, k, norm_kin_data_dict, groups, planes, split):
        #Normative Data
        normdf = pd.DataFrame()
        for side in groups:
            for angle in planes:
                normdf[(side + '_' + angle)] = norm_kin_data_dict[side][angle]['avg_angle']
        normdf = normdf[0:100]
    
        #IMU data
        points = 100
        imudf = pd.DataFrame()
        for side in groups:
            for angle in planes:
                splits = k.export_split(split, side, angle, 'angle', 'degrees', points)
                if splits:
                    df_imu = pd.DataFrame(splits)
                    df_imu.columns = ['start','duration'] + list(range(0,points))
                    df_imu = df_imu.drop(columns = ['start', 'duration'])
                    avg = df_imu.mean(axis=0)
                    imudf[(side + '_' + angle)] = avg
    
        #Normative Score
        angles, sides, scoresx, scoresy = [],[],[],[]
        for side in groups:
            for angle in planes:
                name = (side + '_' + angle)
                d = cdist_dtw(np.squeeze(np.array(normdf[name])), np.squeeze(np.array(imudf[name])))
                du = (1-(d/2000))*100
                if du < 0:
                    du = 0
                r,p = stats.pearsonr(np.squeeze(np.array(normdf[name])), np.squeeze(np.array(imudf[name])))
                angles.append(angle)
                sides.append(side)
                scoresx.append(abs(r)*100)
                scoresy.append(du)
    
    
        scoretable = pd.DataFrame()
        scoretable['Joint'] = angles
        scoretable['Side'] = sides
        scoretable['Normative X Score'] = scoresx
        scoretable['Normative Y Score'] = scoresy
    
    
        return scoretable
    

    def similarityscore(self, k, groups, planes, split):
        
        #IMU data
        points = 100
        angles, sides, scores = [],[],[]
        for side in groups:
            for angle in planes:
                splits = k.export_split(split, side, angle, 'angle', 'degrees', points)
                if splits:
                    df_imu = pd.DataFrame(splits)
                    df_imu.columns = ['start','duration'] + list(range(0,points))
                    df_imu = df_imu.drop(columns = ['start', 'duration'])
                    std = df_imu.std(axis=1)
                    sd_score = np.std(std)
                    sim_score = (1 - (((sd_score - 0) * (1 - 0)) / (2 - 0)) + 0)*100 #Assumes maximum mean SD possible is 2 degrees
                    angles.append(angle)
                    sides.append(side)
                    scores.append(sim_score)
        
        scoretable = pd.DataFrame()  
        scoretable['Joint'] = angles
        scoretable['Side'] = sides
        scoretable['Consistency Score'] = scores 
        
        return scoretable
        
    def symmetryscore(self, k, groups, planes, split):
    
        #IMU data
        points = 100
        listnames, scoresx, scoresy = [],[],[]
        if len(groups) < 2:
            for angle in planes:
                listnames.append(angle)
            scoretable = pd.DataFrame()
            scoretable['Joint'] = listnames
            scoretable['Symmetry Score'] = 'NaN'
            print('Need both left and right sides for symmetry calculations.')
            return scoretable
        else:
            for angle in planes:
                imudf = pd.DataFrame()
                for side in groups:
                    splits = k.export_split(split, side, angle, 'angle', 'degrees', points)
                    if splits:
                        df_imu = pd.DataFrame(splits)
                        df_imu.columns = ['start','duration'] + list(range(0,points))
                        df_imu = df_imu.drop(columns = ['start', 'duration'])
                        avg = df_imu.mean(axis=0)
                        imudf[(side)] = avg
    
                d = cdist_dtw(np.squeeze(np.array(imudf['right'])), np.squeeze(np.array(imudf['left'])))
                du = (1-(d/1000))*100
                if du < 0:
                    du = 0
                r,p = stats.pearsonr(np.squeeze(np.array(imudf['right'])), np.squeeze(np.array(imudf['left'])))
                listnames.append(angle)
                scoresx.append(abs(r)*100)
                scoresy.append(du)
    
            scoretable = pd.DataFrame()
            scoretable['Joint'] = listnames
            scoretable['Symmetry X Score'] = scoresx
            scoretable['Symmetry Y Score'] = scoresy
    
            return scoretable
    
    def gaitscore(self, k, groups, planes, split): 
        #IMU data
        points = 100
        listnames, max_mean_score, max_std_score, maxc_mean_score, maxc_std_score, min_mean_score, min_std_score, minc_mean_score, minc_std_score, rom_mean_score, rom_std_score, sides, angles = [],[],[],[],[],[],[],[],[],[],[],[],[]

        for side in groups:
            for angle in planes:
                splits = k.export_split(split, side, angle, 'angle', 'degrees', points)
                if splits:
                    df_imu = pd.DataFrame(splits)
                    df_imu.columns = ['start','duration'] + list(range(0,points))
                    df_imu = df_imu.drop(columns = ['start', 'duration'])

                    maxangle, minangle, maxcycle, mincycle, rom = [], [], [], [], []
                    for row in range(0, len(df_imu)):
                        line = df_imu.iloc[row][51:]
                        maxangle.append(np.max(line))
                        minangle.append(np.min(line))
                        maxcycle.append(line[line == np.max(line)].index[0])
                        mincycle.append(line[line == np.min(line)].index[0])
                        rom.append(abs(np.max(line) - np.min(line)))

                    listnames.append((side + '_' + angle))
                    sides.append(side)
                    angles.append(angle)
                    max_mean_score.append(np.mean(maxangle))
                    max_std_score.append(np.std(maxangle))
                    min_mean_score.append(np.mean(minangle))
                    min_std_score.append(np.std(minangle))
                    maxc_mean_score.append(np.mean(maxcycle))
                    maxc_std_score.append(np.std(maxcycle))
                    minc_mean_score.append(np.mean(mincycle))
                    minc_std_score.append(np.std(mincycle))
                    rom_mean_score.append(np.mean(rom))
                    rom_std_score.append(np.std(rom))


        scoretable = pd.DataFrame()  
        scoretable['Joint'] = listnames
        scoretable['Angle'] = angles
        scoretable['Side'] = sides
        scoretable['Max_Mean'] = max_mean_score
        scoretable['Max_Std'] = max_std_score
        scoretable['Min_Mean'] = min_mean_score
        scoretable['Min_Std'] = min_std_score

        scoretable['Cycle_Max_Mean'] = maxc_mean_score
        scoretable['Cycle_Max_Std'] = maxc_std_score
        scoretable['Cycle_Min_Mean'] = minc_mean_score
        scoretable['Cycle_Min_Std'] = minc_std_score
        
        scoretable['ROM_Mean'] = rom_mean_score
        scoretable['ROM_Std'] = rom_std_score
        
        return scoretable
    
    def stat_tests(self, k, groups, planes, split): 
        #IMU data
        points = 100
        angles, sides, max_full, min_full, maxc_full, minc_full, rom_full = [],[],[],[],[],[],[]
        for side in groups:
            for angle in planes:
                splits = k.export_split(split, side, angle, 'angle', 'degrees', points)
                if splits:
                    df_imu = pd.DataFrame(splits)
                    df_imu.columns = ['start','duration'] + list(range(0,points))
                    df_imu = df_imu.drop(columns = ['start', 'duration'])

                    maxangle, minangle, maxcycle, mincycle, rom = [], [], [], [], []
                    for row in range(0, len(df_imu)):
                        line = df_imu.iloc[row]
                        maxangle.append(np.max(line))
                        minangle.append(np.min(line))
                        maxcycle.append(line[line == np.max(line)].index[0])
                        mincycle.append(line[line == np.min(line)].index[0])
                        rom.append(abs(np.max(line)) - abs(np.min(line)))

                    angles.append(angle)
                    sides.append(side)
                    max_full.append([maxangle])
                    min_full.append([minangle])
                    maxc_full.append([maxcycle])
                    minc_full.append([mincycle])
                    rom_full.append([rom])


                scoretable = pd.DataFrame()  
                scoretable['Angle'] = angles
                scoretable['Side'] = sides
                scoretable['Max_Full'] = max_full
                scoretable['Min_Full'] = min_full
                scoretable['MaxC_Full'] = maxc_full
                scoretable['MinC_Full'] = minc_full
                scoretable['ROM_Full'] = rom_full
        
        listnames, pvals = [],[]
        for c in ['Max_Full', 'Min_Full', 'MaxC_Full', 'MinC_Full', 'ROM_Full']:
            for a in scoretable.Angle.unique():
                right = scoretable.loc[(scoretable['Angle'] == a) & (scoretable['Side'] == 'right'), c].values[0]
                left =  scoretable.loc[(scoretable['Angle'] == a) & (scoretable['Side'] == 'left'), c].values[0]
                s,p = ttest_ind(np.squeeze(right),np.squeeze(left), equal_var=False)
                listnames.append(a+'_'+c)
                pvals.append(p)

        outtable = pd.DataFrame()
        outtable['Score'] = listnames
        outtable['P-value'] = pvals
        
        return outtable

    
    def asi(self,a,b):
        si = (100 - (abs((a-b)/(0.5*(a+b)))*100))
        if si < 0:
            si = 0
        return si
    
    
    def normgaitscore(self, gaitscores, norm_kin_data_dict, groups, planes):
        normdf = pd.DataFrame()
        angles, sides, Nmax, Nmin, NmaxC, NminC, NROM = [],[],[],[],[],[],[]
        for side in groups:
            for angle in planes:
                normdf[(side + '_' + angle)] = norm_kin_data_dict[side][angle]['avg_angle']
                angles.append(angle)
                sides.append(side)
                Nmax.append(round(normdf[(side + '_' + angle)][51:].max(),2))
                NmaxC.append(normdf[(side + '_' + angle)][normdf[(side + '_' + angle)] == normdf[(side + '_' + angle)][51:].max()].index[0])
                Nmin.append(round(normdf[(side + '_' + angle)][51:].min(),2))
                NminC.append(normdf[(side + '_' + angle)][normdf[(side + '_' + angle)] == normdf[(side + '_' + angle)][51:].min()].index[0])
                NROM.append(round(abs((normdf[(side + '_' + angle)][51:].max()) - (normdf[(side + '_' + angle)][51:].min())),2))


        scoretable = pd.DataFrame()  
        scoretable['Joint'] = angles
        scoretable['Side'] = sides
        scoretable['Normative Max.'] = Nmax
        scoretable['Participant Max.'] = round(gaitscores['Max_Mean'],2)
        scoretable['Compare Max. (%)'] = round(scoretable.apply(lambda row : self.asi(row['Normative Max.'], row['Participant Max.']), axis = 1),2)
        scoretable['Normative Min.'] = Nmin
        scoretable['Participant Min.'] = round(gaitscores['Min_Mean'],2)
        scoretable['Compare Min. (%)'] = round(scoretable.apply(lambda row : self.asi(row['Normative Min.'], row['Participant Min.']), axis = 1),2)
        scoretable['Normative % Cycle Max.'] = NmaxC
        scoretable['Participant % Cycle Max.'] = round(gaitscores['Cycle_Max_Mean'],2)
        scoretable['Compare % Cycle Max. (%)'] = round(scoretable.apply(lambda row : self.asi(row['Normative % Cycle Max.'], row['Participant % Cycle Max.']), axis = 1),2)
        scoretable['Normative % Cycle Min.'] = NminC
        scoretable['Participant % Cycle Min.'] = round(gaitscores['Cycle_Min_Mean'],2)
        scoretable['Compare % Cycle Min. (%)'] = round(scoretable.apply(lambda row : self.asi(row['Normative % Cycle Min.'], row['Participant % Cycle Min.']), axis = 1),2)
        scoretable['Normative ROM'] = NROM
        scoretable['Participant ROM'] = round(gaitscores['ROM_Mean'],2)
        scoretable['Compare ROM (%)'] = round(scoretable.apply(lambda row : self.asi(row['Normative ROM'], row['Participant ROM']), axis = 1),2)

        return scoretable
        
        
    
    ############
    # plotting #
    ############
    def plot_apply_axes(self, axs, xaxis, yaxis):

        if 'lim' in xaxis:
            axs.set_xlim(xaxis['lim'])
        if 'format' in xaxis:
            axs.xaxis.set_major_formatter(xaxis['format'])
        if 'label' in xaxis:
            axs.set_xlabel(xaxis['label'])

        if 'lim' in yaxis:
            axs.set_ylim(yaxis['lim'])
        if 'format' in yaxis:
            axs.yaxis.set_major_formatter(yaxis['format'])
        if 'label' in yaxis:
            axs.set_ylabel(yaxis['label'])

    def plot_ind_on_axis(self, axs, ind, title, xaxis={}, yaxis={}, presentation={}):
        # plot individual step data (kinematics or EMG)
        legend = []

        # colormap of plot lines and step label
        step_c = plt.cm.jet(np.linspace(0,1,len(ind)))

        # plot of indivdual cycles
        for (step, (time_interp, step_interp)) in enumerate(ind):
            axs.plot(time_interp, step_interp, color=step_c[step])

            # create axes
        self.plot_apply_axes(axs, xaxis, yaxis)

        # plot step labels
        if len(ind) > 20:
            label_count = 20
            step_skip = math.ceil(len(ind) / label_count)
        else:
            label_count = len(ind)
            step_skip = 1
        yax = axs.get_ylim()
        ty = yax[0]
        tys = (yax[1]-yax[0])/label_count
        for (step, (time_interp, step_interp)) in enumerate(ind):
             if step % step_skip == 0:
                axs.text(1.02, ty, f'{step}', c=step_c[step], fontsize=8)
                ty = ty + tys # advance y-position of next step label

        if presentation.get('legend') != 'off':
            axs.legend(legend, frameon=False, loc=presentation.get('legend', 1)) #'best'))
        if presentation.get('title') != 'off':
            axs.set_title(title)
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)


    def plot_splits(self, split_name, norm_kin_data=None, norm_emg_data=None,
                    groups=None, positions=None, streams=None, component=None, rms_emg_after_seg=None,
                    target_length=2500, width=plot_width, height=plot_height, plot_ind=True,
                    sheight=None, xaxis={}, yaxis={}, markers={}, presentation={}, config=None, hlines=[0]):

        if presentation.get('style'):
            plt.style.use(presentation['style'])

        # make lists
        if type(streams) != list:
            streams = [streams]
        if type(positions) != list:
            positions = [positions]

        # config is for advanced usage specifying different maps for different sides
        # get sizes from 0th group 0th position
        if config is not None:
            groups = list(config.keys())

            num_ind = 0
            if plot_ind:
                for group in groups:
                    positions = list(config[group].keys())
                    for position in positions:
                        streams = list(config[group][position])
                        num_ind = max(num_ind, len(streams))
        else:
            num_ind = len(positions)*len(streams) 

        nrows = 1
        if plot_ind:
             nrows += num_ind

        ncols = len(groups)
        print(f"rows {nrows} cols {ncols} ind {num_ind}")
        fig, axes = plt.subplots(ncols=ncols, nrows=nrows, constrained_layout=True)
        if sheight is not None:
            height = sheight * nrows
        fig.set_size_inches(width, height, forward = False)
        ax = tools.AX(axes, nrows, ncols)

        for col, group in enumerate(groups):
            if not self.splits.get(group, []):
                print(f"no splits for {group}")
                continue
            # plot combined
            axs = ax.axs(0, col)
            handles = []
            legend = []
            individuals = {}

            if config:
                positions = config[group].keys()

            for position in positions:
                if config:
                    streams = sorted(config[group][position])
                for stream in streams:
                    if not self.check(group, position, stream, component):
                        continue
                    if split_name not in self.splits[group]:
                        continue
                    label = f'{position}_{stream}'
                    steps = self.calculate_split(split_name, group, position, stream, component, target_length=target_length, rms_emg_after_seg=rms_emg_after_seg)
                    avg = steps['avg']
                    std = steps['std']
                    x_percent = np.linspace(0,1,num=len(avg)) # x axis range: 0 to 1
                    line, = axs.plot(x_percent, avg)
                    fill = axs.fill_between(x_percent, avg - std, avg + std, alpha=0.5)
                    handles.append(fill)
                    legend.append(label)
                    individuals[label] = steps['ind']

            self.plot_apply_axes(axs, xaxis, yaxis)  

            if group in markers:
                for mark in markers[group]:
                    line = axs.axvline(mark['x'], label=mark['label'], color=mark['color'])
                    if 'stdev' in mark:
                        axs.axvspan(mark['x']-mark['stdev'], mark['x']+mark['stdev'], color=mark['color'], alpha=0.2)
                    handles.append(line)
                    legend.append(mark['label'])
            
            for y in hlines:
                axs.axhline(y=y, linewidth=1)
            
            if presentation.get('legend') != 'off':
                axs.legend(handles, legend, frameon=False, loc=presentation.get('legend', 'best'))
            if presentation.get('title') != 'off':
                axs.set_title(f'{group}')
            axs.spines['top'].set_visible(False)
            axs.spines['right'].set_visible(False)

            # plot individual steps (default = TRUE)
            if plot_ind:
                i = 0 
                for (position, ind) in individuals.items():
                    axs = ax.axs(1+i, col)
                    i += 1
                    self.plot_ind_on_axis(axs, ind, position, xaxis, yaxis, presentation)

                    ######## PLOT NORMATIVE DATA #########
                    # EMG (Lencioni et al. dataset)
                    if norm_emg_data is not None:

                        # match Cionic muscle labeling with normative EMG data labeling
                        ### lower leg (shank)
                        if 'gl' in position: muscle = 'GL'
                        elif 'gm' in position: muscle = 'GM'
                        elif 'ta' in position: muscle = 'TA'
                        ### upper leg (thigh)
                        elif 'rf' in position: muscle = 'RF'
                        elif 'vm' in position: muscle = 'VM' 
                        elif 'vl' in position: muscle = 'VL'
                        elif 'hm' in position: muscle = 'BF'

                        else: continue # if positions are not present, don't plot normative EMG
                            
                        if norm_emg_data['avg'].get(muscle) is None:
                            continue
                        
                        # specify data (all data must be in decimals, not percentage values)
                        x_percent_norm = np.divide(norm_emg_data['gcycle_percent'], 100)
                        emg_norm_avg = np.divide(norm_emg_data['avg'][muscle], 100)
                        emg_norm_std = np.divide(norm_emg_data['std'][muscle], 100)
                        
                        # plot
                        axs.plot(x_percent_norm, emg_norm_avg, c='black',linewidth=5, alpha=0.5)
                        axs.fill_between(x_percent_norm, np.subtract(emg_norm_avg,emg_norm_std), np.add(emg_norm_avg,emg_norm_std), color='lightgrey',alpha=0.5)

                        # label muscle of normative EMG
                        axs.text(0.4, 4.5, f'Grey = normative ({muscle})', fontsize=12)

                    # KINEMATICS
                    if norm_kin_data is not None:
                        pos = position[0:-1-5] # get rid of the '_angles' part of string position

                        x_percent_norm = np.array(norm_kin_data[group][pos]['gait_cycle']) / 100 # values = decimals, labels = %
                        avg_norm = norm_kin_data[group][pos]['avg_angle']
                        std_norm = norm_kin_data[group][pos]['std_angle']

                        axs.plot(x_percent_norm, avg_norm, c='grey')
                        axs.fill_between(x_percent_norm, np.subtract(avg_norm,std_norm), np.add(avg_norm,std_norm), color='lightgrey',alpha=0.5)
        self.save_fig(fig)
        plt.show()

    def plot_split_streams(self, group, stream, components, axs, legend):
        if stream != "splits":
            return False
        comps = components if components else [ k for k in self.splits[group]]
        for component in comps:
            if component in self.splits[group]:
                splits = self.splits[group][component]
                splits_len = len(splits['splits']) - len(splits['skips'])
                print(f"{splits_len} splits for {group} {component}")
                x = splits['streams']['elapsed_s']
                y = splits['streams']['peak']
                axs.plot(x, y)
                legend.append(f"{group}_{component}_splits")
        return True


    def plot(self, groups, positions=None, streams=None, components=None, width=plot_width, height=plot_height, offset=0, hlines=None, plot_time_ranges=True):

        fig, axes = plt.subplots(ncols=1, nrows=1, constrained_layout=True)
        fig.set_size_inches(width, height, forward = False)
        ax = tools.AX(axes, 1, 1)
        legend = []
        axs = ax.axs(0, 0) 

        off = 0
        for group in groups:
            if positions is None:
                positions = list(self.groups[group].keys())
            for position in positions:
                if streams is None:
                    streams = list(set(functools.reduce(operator.iconcat, [ list(self.groups[group][pos].keys()) for pos in positions ], [])))
                for stream in streams:
                    if self.plot_split_streams(group, stream, components, axs, legend):
                        continue
                    if not self.check(group, position, stream):
                        continue
                    ndarray = self.groups[group][position][stream]
                    ndkeys = ndarray.dtype.names
                    comps = components if components else [ k for k in ndkeys if k != 'elapsed_s']
                    for component in comps:
                        if component in ndkeys:
                            x = ndarray['elapsed_s']
                            y = ndarray[component] + off
                            axs.plot(x, y)
                            legend.append(f"{position}_{stream}_{component}")
                        off += offset

            if plot_time_ranges:
                if self.time_ranges[group] is None:
                    continue
                for time_range in self.time_ranges[group]:
                    assert len(time_range) == 2
                    start_time_range, stop_time_range = time_range
                    axs.axvspan(start_time_range, stop_time_range, alpha=0.05)

        if hlines is not None:
            for y in hlines:
                axs.axhline(y)

        axs.set_xlabel('elapsed')
        #axs.set_ylabel(component)
        axs.legend(legend, frameon=False, loc='best')
        #axs.set_title(f'{component}')
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        plt.show()

    def plot_joint_angles(self, angles, width=plot_width, height=plot_height):
        nrows = len(angles)
        ncols = len(angles[0])
        fig, axes = plt.subplots(ncols=ncols, nrows=nrows, constrained_layout=True)
        fig.set_size_inches(width, height, forward = False)
        ax = tools.AX(axes, nrows, ncols)

        for (r, row) in enumerate(angles):
            for (c, col) in enumerate(row):
                (group, angle, ytop, ybottom) = col
                legend = []
                axs = ax.axs(r, c)

                self.plot_splits(axs, group, angle, ylim=(ybottom, ytop))

                axs.set_title(f'{group} {angle}')
                axs.set_xlabel('% cycle')
                axs.set_ylabel('degrees')
                axs.legend(legend, frameon=False, loc='best')
                axs.spines['top'].set_visible(False) 
                axs.spines['right'].set_visible(False)

    def plot_muscle_wheel(self, split, mconfig, step, mult, width, height):

        fig, axes = plt.subplots(ncols=2, nrows=1, constrained_layout=True, subplot_kw={'polar':True})
        fig.set_size_inches(width, height, forward = False)
        ax = tools.AX(axes, 1, 2)

        data_theta_rad = []
        for i in range(0,360,1):
            data_theta_rad.append(float(i)*np.pi/180.0)

        for i, side in enumerate(['left', 'right']):
            axs = ax.axs(0, i)
            axs.set_title(side, va='bottom')
            axs.set_theta_zero_location("N")
            axs.set_theta_direction(-1)
            axs.set_yticklabels([])
            legend = []
            #axs.set_rgrids(radii, labels=labels)
            offset = 0
            for (label, position, stream) in mconfig[side]:
                offset += step

                if self.check(side, position, stream) is False:
                    continue

                s = self.calculate_split(split, side, position, stream, 'emg', 360)
                data = s['avg']*mult+offset

                #ax.plot(data_theta_rad, data, color='r', linewidth=3)
                axs.fill_between(data_theta_rad, offset, data, alpha=1.0)
                legend.append(label)

            axs.grid(True)
            axs.legend(legend, frameon=False, loc='best')

        self.save_fig(fig)

    ###########################
    # Quaternion Computations #
    ###########################

    def to_2d_array(self, arr):
        """ Converts Cionic custom array to 2d array. """
        arr2d = np.zeros([arr.shape[0], len(arr[0])])
        for idx, tup in enumerate(arr):
            arr2d[idx, :] = list(tup)
        return arr2d


    def load_calibration(self, group, position, stream, calibration):
        # TODO : see if we can avoid eval
        if calibration != "" and stream == "fquat":
            # fquat calibration is 5 floats representing upright orientation difference quaternion and xoffset
            self.calibrations[group][position][stream] = struct.unpack("<5f", eval(calibration))
        elif calibration != "" and stream == "emg":
            # emg calibration is a float value per channel impedances
            cal = eval(calibration)
            form = f"<{int(len(cal)/4)}f"
            self.calibrations[group][position][stream] = struct.unpack(form, cal)
        else:
            return None

        return np.array(self.calibrations[group][position][stream])


    def orientations_for_single_stream(self, data, calibration=None):
        """ Computes orientations from a single stream of quats in data. """
        ijkr_components = [name for name in data.dtype.names if name != "elapsed_s"]
        quaternions = self.to_2d_array(data[ijkr_components])
        sensor = Rotation.from_quat(quaternions)

        if calibration is not None:
            upright = Rotation.from_quat(calibration[0:4])
            forward = Rotation.from_quat([ 0, 0, np.sin(calibration[4]/2), np.cos(calibration[4]/2) ])
            norm = Rotation.from_quat([ 0, 0, np.sin(-calibration[4]/2), np.cos(-calibration[4]/2) ])
            orientation = norm * sensor * upright * forward
        else:
            orientation = sensor

        orientation_dict = {
            'orientation': orientation,
            'elapsed_s': data["elapsed_s"]
        }
        return orientation_dict


    def calculate_angle(self, q1, q2, seq):
        """ Calculates the relative angle between two rotations and returns Euler. """
        relative = q1.inv() * q2
        return relative.as_euler(seq, degrees=True)


    def get_data_array_and_calibration(self, group, position, stream):
        """ Returns the data array and calibration for a group, position, and stream. """
        data = self.groups[group][position][stream]
        if stream in self.calibrations[group][position]:
            calibration = self.calibrations[group][position][stream]
        else:
            calibration = None
        return data, calibration


    def get_synthetic_interpolated_array(self, arr_a, arr_b):
        """ Returns an interpolated sythetic array given two arrays. Used specifically in the context of time arrays. """
        sampling_rate = np.mean([np.median(np.diff(arr_a)), np.median(np.diff(arr_b))])
        min_timestamp = max(arr_a.min(), arr_b.min())
        max_timestamp = min(arr_a.max(), arr_b.max())
        elapsed_s_interp = np.arange(min_timestamp, max_timestamp, sampling_rate)
        # Remove edge case arrays created with values exceeding min_timestamp, max_timestamp. Not sure why this happens, but this line seems to fix the issue.
        elapsed_s_interp = elapsed_s_interp[(elapsed_s_interp <= max_timestamp) & (elapsed_s_interp >= min_timestamp)]
        return elapsed_s_interp


    def get_nonincreasing_indices(self, arr):
        """
        Returns all indices where an array is non increasing.
        This is used to remove nonincreasing timestamps for Slerp.
        """
        arr = np.maximum.accumulate(arr)
        ind = np.where(np.sign(np.diff(arr)) != 1)[0] + 1
        return ind


    def remove_nonincreasing_entries(self, arr):
        """ Removes all entries in an array with non increasing timestamps. """
        ind = self.get_nonincreasing_indices(arr["elapsed_s"])
        mask = np.ones(arr.size, dtype=bool)
        mask[ind] = False
        arr = arr[mask]
        return arr


    def calculate_joint_angle(self, group, position_a, position_b, stream):
        """ Calculates a joint angle in Eulers between position_a and positions_b give in quats"""
        # determine streams from config
        data_a, calibration_a = self.get_data_array_and_calibration(group, position_a, stream)
        data_b, calibration_b = self.get_data_array_and_calibration(group, position_b, stream)

        # remove samples with nonincreasing timestamps.
        n_a = data_a.shape[0]
        n_b = data_b.shape[0]
        data_a = self.remove_nonincreasing_entries(data_a)
        data_b = self.remove_nonincreasing_entries(data_b)

        if n_a != data_a.shape[0]:
            print(f"{position_a} {stream}: {n_a - data_a.shape[0]} nonincreasing samples removed (of {data_a.shape[0]} total samples).")
        if n_b != data_b.shape[0]:
            print(f"{position_b} {stream}: {n_b - data_b.shape[0]} nonincreasing samples removed (of {data_b.shape[0]} total samples).")

        # Get rotations for each stream.
        orientations_a = self.orientations_for_single_stream(data_a, calibration=calibration_a) # dict with many Rotations and time array
        orientations_b = self.orientations_for_single_stream(data_b, calibration=calibration_b) # dict with many Rotations and time array

        elapsed_s_interp = self.get_synthetic_interpolated_array(orientations_a["elapsed_s"], orientations_b["elapsed_s"])

        # Get interpolated rotations for each stream.
        slerp_a = Slerp(orientations_a["elapsed_s"], orientations_a["orientation"])
        orientations_a_interp = np.asarray(slerp_a(elapsed_s_interp))

        slerp_b = Slerp(orientations_b["elapsed_s"], orientations_b["orientation"])
        orientations_b_interp = np.asarray(slerp_b(elapsed_s_interp))

        # Calculate relative Euler angle between two Rotations.
        euler_stream = []
        for idx in range(elapsed_s_interp.shape[0]):
            q_a = orientations_a_interp[idx]
            q_b = orientations_b_interp[idx]
            euler_sample = tuple(self.calculate_angle(q_a, q_b, 'xyz')) + (elapsed_s_interp[idx], )
            euler_stream.append(euler_sample)

        eulers = np.array(euler_stream, dtype={'names': ('x', 'y', 'z', 'elapsed_s'), 'formats':('f8', 'f8', 'f8', 'f8')})
        return eulers


    def create_upright_stream(self, arr):
        upright = []
        for x in arr:
            upright.append((0,0,0,1, x[4]))
        return np.array(upright, dtype=arr.dtype)        


    def create_upright(self, group, angles, stream):
        for joint, angles in angles.items():
            (position_a, position_b) = joint
            if self.check(group, position_a, stream):
                self.groups[group]['upright'][stream] = self.create_upright_stream(self.groups[group][position_a][stream])
                return
            if self.check(group, position_b, stream):
                self.groups[group]['upright'][stream] = self.create_upright_stream(self.groups[group][position_b][stream])
                return


    def calculate_joint_angles(self, group, angles, stream='fquat', neutral_offsets={}):
        """ Calculates joint angles based on quat streams of positions and assigns Eulers to self.groups. """

        # add an upright vector to the group
        # match in length to the first rotation stream
        self.create_upright(group, angles, stream)

        print('\nComputing joint angles...')
        for joint, angles in angles.items():

            (position_a, position_b) = joint

            print(f"between {position_a} and {position_b} ({stream} stream)")

            if not self.check(group, position_a, stream):
                continue
            if not self.check(group, position_b, stream):
                continue

            computed = self.calculate_joint_angle(group, position_a, position_b, stream)
            for axis, angle in angles.items():
                if axis[0] == '-':
                    arr = np.array(computed[[axis[1], "elapsed_s"]], dtype={'names': ('degrees', 'elapsed_s'), 'formats':('f8', 'f8')})
                    arr["degrees"] = - arr["degrees"]
                else:
                    arr = np.array(computed[[axis, "elapsed_s"]], dtype={'names': ('degrees', 'elapsed_s'), 'formats':('f8', 'f8')})

                # calculate pelvis, hip, knee, ankle angles (°)
                self.groups[group][angle]['angle'] = np.array(arr, dtype={'names': ('degrees', 'elapsed_s'),
                                                                                     'formats':('f8', 'f8')})
                # offset angles by any passed neutral_offset default to 0
                offset = neutral_offsets.get(group, {}).get(angle, 0)
                self.groups[group][angle]['angle']['degrees'] += offset


    def calculate_limb_angles(self, group, positions, stream='fquat'):
        """ Calculates limb angles based on quat streams of positions and assigns Eulers to self.groups. """
        print('\nComputing limb angles...')
        for position in positions:

            if not self.check(group, position, stream):
                continue

            print(f"for {position} ({stream} stream)")

            data, calibration = self.get_data_array_and_calibration(group, position, stream)

            orientations = self.orientations_for_single_stream(data, calibration=calibration)
            eulers = orientations["orientation"].as_euler('xyz', degrees=True)
            elapsed_s = orientations["elapsed_s"]

            euler_stream = []
            for idx in range(eulers.shape[0]):
                euler_sample = tuple(eulers[idx, :]) + (elapsed_s[idx], )
                euler_stream.append(euler_sample)

            self.groups[group][position]['euler'] = np.array(euler_stream,
                                                             dtype={'names': ('x', 'y', 'z', 'elapsed_s'),
                                                                    'formats':('f8', 'f8', 'f8', 'f8')})


    ####################
    # EMG Computations #
    ####################

    def emg_fields(self, group, position, stream="emg"):
        if stream in self.groups[group][position]:
            return [ fld for fld in self.groups[group][position][stream].dtype.fields if fld != 'elapsed_s' ]
        else:
            return []

    def load_channel_pos(self, group, position, stream, channels):
        if channels != "":
            self.channels[group][position][stream] = channels.split(" ")

    def calculate_emgs(self, group, positions, emg_params, stream="emg"):

        rmsw = emg_params.get('rms_window')
        filt = emg_params.get('filter')
        norm = emg_params.get('normalize')

        for position in positions:
            if not self.check(group, position, stream):
                continue

            emg = self.groups[group][position][stream]
            try:
                regs = self.regs[group][position][stream]
            except:
                regs = None
            emg_params['sampling_rate'] = tools.regs_sampling_rate(regs)

            fields = self.emg_fields(group, position, stream)
            if stream in self.channels[group][position]:
                channel_map = self.channels[group][position][stream]
                # Address init.f misspecification where more channels are streaming than channel names provided.
                if len(fields) > len(channel_map):
                    channel_map = fields
            else:
                channel_map = fields

            for i, field in enumerate(fields):
                cpos = channel_map[i]
                print(f"computing emg for {position} {field} at {cpos}")
                if regs is None:
                    output = emg[field]
                else:
                    output = tools.regs_convert_uV(emg[field], regs, field)
                elapsed = emg['elapsed_s']

                if filt:
                    output = filt(output, emg_params)

                if rmsw:
                    output = tools.moving_avg_rms(output, window_size = rmsw)
                    start_idx = int((rmsw - 1) // 2)
                    end_idx = - int(rmsw // 2)
                    elapsed = elapsed[start_idx: end_idx]

                if norm:
                    output = output / norm(output)

                arr = np.array(list(zip(output, elapsed)), dtype={'names': ('emg', 'elapsed_s'),
                                                                  'formats':('f8', 'f8')})
                self.groups[group][position][cpos] = arr

    def calculate_emgs_remap(self, group, positions, emg_params, stream="emg"):

        rmsw = emg_params.get('rms_window')
        filt = emg_params.get('filter')
        norm = emg_params.get('normalize')

        for (position, config) in positions.items():
            if not self.check(group, position, stream):
                continue

            emg = self.groups[group][position][stream]
            regs = self.regs[group][position][stream]
            emg_params['sampling_rate'] = tools.regs_sampling_rate(regs)

            for (component, remap) in config.items():
                if component not in emg.dtype.fields:
                    continue

                output = tools.regs_convert_uV(emg[component], regs, component)
                elapsed = emg['elapsed_s']

                if filt:
                    output = filt(output, emg_params)

                if rmsw:
                    output = tools.moving_avg_rms(output, window_size = rmsw)
                    elapsed = elapsed[rmsw-1:]

                if norm:
                    output = output / norm(output)

                arr = np.array(list(zip(output, elapsed)), dtype={'names': ('emg', 'elapsed_s'),
                                                                  'formats':('f8', 'f8')})
                self.groups[group][remap][stream] = arr


    ############
    # Pressure #
    ############

    def calculate_pressures(self, group, positions, stream="ladc", stream_out="pressure", component_out="mv", pressure_params={}):

        rmsw = pressure_params.get('rms_window')
        filt = pressure_params.get('filter')
        norm = pressure_params.get('normalize')
        offset = pressure_params.get('offset')

        for position in positions:

            if not self.check(group, position, stream):
                continue

            emg = self.groups[group][position][stream]
            try:
                regs = self.regs[group][position][stream]
            except Exception as e:
                regs = None

            fields = self.emg_fields(group, position, stream)
            if stream in self.channels[group][position]:
                channel_map = [ ch for ch in self.channels[group][position][stream] if ch ]
                # protect against bad mapping
                if len(channel_map) != len(fields):
                    logging.error(f"CHANPOS {channel_map} does not match FIELDS {fields}")
                    channel_map = fields
            else:
                channel_map = fields

            for i, field in enumerate(fields):
                cpos = channel_map[i]
                print(f"computing pressure for {position} {field} at {cpos}")
                if regs is None:
                    output = emg[field]
                else:
                    output = tools.regs_convert_uV(emg[field], regs, field)
                elapsed = emg['elapsed_s']

                if offset:
                    output += offset

                if filt:
                    output = filt(output, pressure_params)

                if rmsw:
                    output = tools.moving_avg_rms(output, window_size = rmsw)
                    elapsed = elapsed[rmsw-1:]

                if norm:
                    output =  output / norm(output)

                self.groups[group][position][cpos] = np.array(list(zip(output, elapsed)), 
                                                             dtype={'names': (component_out, 'elapsed_s'),
                                                                    'formats':('f8', 'f8')})

    ############
    # Digipots #
    ############

    def calculate_fes(self, group, positions, stream_in="digip", stream_out="fes", component_out="perc", scale=1):
        """
        remap the components of the pressure sensor to their own locations
        """ 
        for (position, component_map) in positions.items():
            for (component, remap) in component_map.items():
                if not self.check(group, position, stream_in):
                    continue

                streamvals = self.groups[group][position][stream_in]
                # invert everything
                normalized = 200-streamvals[component]
                normalized = np.maximum(normalized, np.zeros_like(normalized))
                elapsed = streamvals['elapsed_s']

                # there must be a better way
                last = 0
                vals = []
                times = []
                for idx in np.where(np.diff(elapsed) > 0.02)[0]:
                    if idx <= last+1:
                        continue
                    t_prior = elapsed[last:idx]
                    v_prior = normalized[last:idx]
                    samp = np.mean(np.diff(t_prior))
                    pts = (elapsed[idx+1]-elapsed[idx])/samp
                    t_fill = np.linspace(elapsed[idx], elapsed[idx+1], pts)
                    v_fill = np.zeros_like(t_fill)
                    times = np.concatenate((times, t_prior, t_fill), axis=0)
                    vals = np.concatenate((vals, v_prior, v_fill), axis=0)
                    last = idx+1

                self.groups[group][remap][stream_out] = np.array(list(zip(vals, times)), 
                                                                 dtype={'names': (component_out, 'elapsed_s'),
                                                                        'formats':('f8', 'f8')})

    def calculate_deveuler(self, group, positions, stream_in="euler", stream_out="angle", component_out="degrees", scale=1):
        """
        remap the components of the pressure sensor to their own locations
        """ 
        for (position, component_map) in positions.items():
            for (component, remap) in component_map.items():
                if not self.check(group, position, stream_in):
                    continue

                # TODO : any normalization needed for pressure
                streamvals = self.groups[group][position][stream_in]
                normalized = np.degrees(streamvals[component])
                elapsed = streamvals['elapsed_s']
                self.groups[group][remap][stream_out] = np.array(list(zip(normalized, elapsed)), 
                                                                 dtype={'names': (component_out, 'elapsed_s'),
                                                                        'formats':('f8', 'f8')})

    #################
    # stride length #
    #################
    
    def next_split_index(self, index, splits):
        while True:
            index += 1
            if index >= len(splits['splits']):
                return None
            if index in splits['skips']:
                continue
            return index
    
    def signal_at_splits(self, splits, group, position, stream, component, ts="elapsed_s"):
        points = []
        if not self.check(group, position, stream, component):
            return points
    
        vals = self.groups[group][position][stream]
        split_index = self.next_split_index(0, splits)
    
        for val in vals:
            if split_index is None:
                break
            split = splits['splits'][split_index]
            
            if val[ts] >= split[0]:
                points.append((split[0], val[ts], val[component], split_index))
                split_index = self.next_split_index(split_index, splits)
    
        return points
    
    def add_knee_angles(self, splits, side, signals, time_key, entry_key):
        knee_angles = self.signal_at_splits(splits, side, 'knee_flexion', 'angle', 'degrees')
        knee_angle_d = {}
        for ka in knee_angles:
            knee_angle_d[ka[0]] = ka[2]
            
        for signal in signals:
            signal[entry_key] = knee_angle_d[signal[time_key]]
        
    def gait_cycles(self, side, thigh, shank):
        """ calculate triangulated hip translation during stance (heel_strike to toe_off)
            and foot translation during swing (toe_off to heel_strike)
        """
        cycles = []
        toe_offs = self.splits[side].get('toe_off')
        heel_strikes = self.splits[side].get('heel_strike')
    
        if not toe_offs or not heel_strikes:
            print(f"insufficient splits for {side}")
            return cycles
    
        toe_off_shank = self.signal_at_splits(toe_offs, side, shank, 'euler', 'x')
        toe_off_thigh = self.signal_at_splits(toe_offs, side, thigh, 'euler', 'x')
    
        heel_strike_shank = self.signal_at_splits(heel_strikes, side, shank, 'euler', 'x')
        heel_strike_thigh = self.signal_at_splits(heel_strikes, side, thigh, 'euler', 'x')
    
        #first_toe_off = toe_off_shank[0][0]
        #first_heel_strike = heel_strike_shank[0][0]
    
        # construct stance entries - heel strike to toe off
        stances = []
        toe_index = 0
        for i, hs_shank in enumerate(heel_strike_shank):
            # get the next toe index
            while toe_index < len(toe_off_shank) and toe_off_shank[toe_index][0] < hs_shank[0]:
                toe_index += 1
            if toe_index >= len(toe_off_shank):
                break
                
            hs_thigh = heel_strike_thigh[i]
            to_shank = toe_off_shank[toe_index]
            to_thigh = toe_off_thigh[toe_index]
            stances.append({
                'hs_ts' : hs_shank[0], 
                'to_ts' : to_shank[0], 
                'hs_sx' : hs_shank[2], 
                'hs_tx' : hs_thigh[2], 
                'to_sx' : to_shank[2], 
                'to_tx' : to_thigh[2],
                'hs_in' : hs_shank[3]
            })
        self.add_knee_angles(heel_strikes, side, stances, 'hs_ts', 'hs_ka')
        self.add_knee_angles(toe_offs, side, stances, 'to_ts', 'to_ka')
            
        # construct swing entries - toe off to heel strike
        swings = []
        heel_index = 0
        for i, to_shank in enumerate(toe_off_shank):
            # get the next heel index
            while heel_index < len(heel_strike_shank) and heel_strike_shank[heel_index][0] < to_shank[0]:
                heel_index += 1
            if heel_index >= len(heel_strike_shank):
                break

            to_thigh = toe_off_thigh[i]
            hs_shank = heel_strike_shank[heel_index]
            hs_thigh = heel_strike_thigh[heel_index]
            swings.append({
                'hs_ts' : hs_shank[0], 
                'to_ts' : to_shank[0], 
                'hs_sx' : hs_shank[2], 
                'hs_tx' : hs_thigh[2], 
                'to_sx' : to_shank[2], 
                'to_tx' : to_thigh[2],
                'hs_in' : hs_shank[3]
            })
        self.add_knee_angles(heel_strikes, side, swings, 'hs_ts', 'hs_ka')
        self.add_knee_angles(toe_offs, side, swings, 'to_ts', 'to_ka')
    
        # construct cycles - heel strike to heel strike
        swing_index = 0
        for i, stance in enumerate(stances):
            # get to the matching swing
            while swing_index < len(swings) and swings[swing_index]['to_ts'] < stance['to_ts']:
                swing_index += 1
            if swing_index >= len(swings):
                break
            swing = swings[swing_index]

            # possible to have fast forwarded to a swing that is beyond the 
            if swing['to_ts'] > stance['to_ts']:
                continue

            # possible that the new swing is not cconsecutive with the last becase of skips
            if (swing['hs_in'] - stance['hs_in']) != 1:
                continue
            
            assert(stance['to_ts'] == swing['to_ts']) # make sure we are properly matched
            cycles.append({
                'cycle_ts' : stance['hs_ts'],
                'cycle_s' : swing['hs_ts'] - stance['hs_ts'],
                'cycle_stance' : stance['to_ts'] - stance['hs_ts'],
                'cycle_swing' : swing['hs_ts'] - swing['to_ts'],
                'stance_hs_sx' : stance['hs_sx'],
                'stance_hs_tx' : stance['hs_tx'],
                'stance_hs_ka' : stance['hs_ka'],
                'to_sx' : stance['to_sx'],
                'to_tx' : stance['to_tx'],
                'to_ka' : stance['to_ka'],
                'swing_hs_sx' : swing['hs_sx'],
                'swing_hs_tx' : swing['hs_tx'],
                'swing_hs_ka' : swing['hs_ka']
            })
    
        return cycles
    
    
    def gait_triangles(self, cycles, thigh_len, shank_len):
        for cycle in cycles:
            # first solve for the distance from the hip to the foot for the three points in the gait cycle
            stance_hs_tri = costri(shank_len, thigh_len, math.radians(180-cycle['stance_hs_ka']))
            to_tri = costri(shank_len, thigh_len, math.radians(180-cycle['to_ka']))
            swing_hs_tri = costri(shank_len, thigh_len, math.radians(180-cycle['swing_hs_ka']))
    
            # save the calculated hip-to-foot lengths from the triangles
            stance_hs_len = stance_hs_tri['c']
            to_len = to_tri['c']
            swing_hs_len = swing_hs_tri['c']
            # print("lengths", stance_hs_len, to_len, swing_hs_len)
            
            # calculate the travel of the shank during the stance phase = stance_hs to toe_off
            # and the resultant angle between the hip-to-foot vectors
            # and the translation of the hip during stance
            stance_shank_angle = math.radians(cycle['stance_hs_sx'] - cycle['to_sx'])
            stance_trans_angle = stance_shank_angle - to_tri['B'] + stance_hs_tri['B']
            stance_trans_tri = costri(stance_hs_len, to_len, stance_trans_angle)
            stance_trans_len = stance_trans_tri['c']
            # print("stance", math.degrees(stance_shank_angle), math.degrees(stance_trans_angle), stance_trans_len)

            # the hip will continue to translate during swing - assume constant speed
            swing_hip_trans = stance_trans_len * cycle['cycle_swing'] / cycle['cycle_stance']
    
            # calculate the travel of the thigh during the swing phase = toe_off to swing_hs
            # and the resultant travel of the foot-to-hip vectors
            # and the translation of the foot during swing
            swing_thigh_angle = math.radians(cycle['swing_hs_tx'] - cycle['to_tx'])
            swing_trans_angle = swing_thigh_angle - swing_hs_tri['A'] + to_tri['A']
            swing_trans_tri = costri(swing_hs_len, to_len, swing_trans_angle)
            swing_trans_len = swing_trans_tri['c']
            # print("swing", math.degrees(swing_thigh_angle), math.degrees(swing_trans_angle), swing_trans_len)
            
            cycle.update({
                'stance_hs_len' : stance_hs_len,
                'to_len' : to_len,
                'swing_hs_len' : swing_hs_len,
                'stance_trans_len' : stance_trans_len,
                'swing_trans_len' : swing_trans_len,
                'swing_hip_trans' : swing_hip_trans,
            })

    def stride_len_simple(self, sdf, field):
        stance_factor = 1.3
        swing_factor = 0
        swing_hip_factor = 0.9
        sdf[field] = (stance_factor*sdf["stance_trans_len"] + swing_factor*sdf["swing_trans_len"] + swing_hip_factor*sdf["swing_hip_trans"])
        
    def stride_len_predict(self, sdf, field):
        """
        fields = [ 'stance_trans_len', 'swing_hip_trans', 'cycle_swing', 'cycle_stance' ]
        coeffs = [ 0.21436976, 2.71453041, -186.91077322, 12.11707543 ]
        intercept = 88.93377637
        
        fields = ['stance_trans_len','swing_trans_len', 'cycle_swing', 'cycle_stance']
        coeffs = [ 1.00014027, 0.69975196, 7.70833072, -55.45432932] 
        intercept = 59.31022309
        
        # pass 1
        fields = [ 'stance_trans_len', 'swing_hip_trans' ,'swing_trans_len', 'cycle_swing', 'cycle_stance' ]
        coeffs = [ 3.49741966e-02, 2.29679101e+00, 6.83320911e-01, -1.67316135e+02, 1.52764733e+01] 
        intercept = 61.51849456
        """
        
        # pass 2 -- 004 Slow 1 and Slow 2 eliminated (note that the coefficients above excluded slow speeds
        fields = [ 'stance_trans_len', 'swing_hip_trans' ,'swing_trans_len', 'cycle_swing', 'cycle_stance' ]
        coeffs = [ 1.11185991e-01,  1.95678223e+00, 6.64357805e-01, -1.20504130e+02, -1.67016521e+00]
        intercept = 66.24007294
        
        sdf[field] = (intercept + 
                      sdf[fields[0]]*coeffs[0] +
                      sdf[fields[1]]*coeffs[1] +
                      sdf[fields[2]]*coeffs[2] +
                      sdf[fields[3]]*coeffs[3] + 
                      sdf[fields[4]]*coeffs[4])


    def gait_metrics(self, sdf, side, metrics, f=np.mean):
        def included(row, gm, m, f):
            sts = row['cycle_ts']
            ets = sts + row['cycle_s']
            arr = gm[(gm['elapsed_s']>=sts) & (gm['elapsed_s']<=ets)]
            return f(arr[m])
            
        limbs = {
            'right' : 'r_thigh',
            'left' : 'l_thigh',
        }
        gm = pd.DataFrame(self.groups[side][limbs[side]]['gait_metrics'])
        for m in metrics:
            sdf[m] = sdf.apply(lambda row: included(row, gm, m, f), axis=1)
            

