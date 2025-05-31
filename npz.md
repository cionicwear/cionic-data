# Cionic NPZ files

Data streams from a Cionic Collection are packaged into an npz file.  

First activate virtual environment:

`python3 -m venv venv`

`source venv/bin/activate`


Let's start by examining an unsegmented npz
```
>>> import numpy as np
>>> npz = np.load('<path_to_npz>')
>>> npz.files
```

This will list the files contained within the npz
```
# summary table
'segments',

# sensor data streams
'SI_10000002_emg', 'SI_10000002_fquat', 'SI_10000002_adcf', 'DC_10000001_fquat',

# algorithms defined in the protocol and their metadata
'DC_10000001_cf_relax', 'DC_10000001_cf_contract', 
'DC_10000001_cf_contract_meta', 'DC_10000001_cf_relax_meta',

# supporting algorithms run by the OS
'DC_10000001_35_0_pk', 'DC_10000001_35_0_tr',
'DC_10000001_35_0_pk_meta', 'DC_10000001_35_0_tr_meta',  

# json data 
'devices.jsonl', 'gwlabels.jsonl', 'segments.jsonl', 'collection.json', 'protocol.json']
```

## Segments
The `segments` file provides a high level table of contents for the npz file
```
>>> import pandas as pd
>>> df = pd.DataFrame(npz['segments'])
>>> df.keys()
Index(['fields', 'struct', 'mode', 'sistreamnum', 'calibration', 'position',
       'buspos', 'device', 'sensor', 'streamnum', 'stream', 'path', 'start_s',
       'end_s', 'duration_s', 'nsamples', 'avg_rate_hz', 'zones', 'chanpos'],
      dtype='object')
      
>>> set(df['stream'])
```

Looking at the available streams we see the following
```
'fquat',             # quaternion data
'emg',               # raw EMG data (requires processing)
'adcf',              # filtered EMG data 1st order diff 
'35_0_pk',           # peak detector running on stream 35 component 0
'35_0_tr',           # trough detector running on stream 35 component 0
'cf_contract',       # cf_contract trigger as specified in the protocol
'cf_relax',          # cf_relax trigger as specified in the protocol
...
```

Now selecting just for quaternion streams we see a stream for `l_thigh` and `l_shank`
```
>>> df[df['stream']=='fquat']

        fields struct mode sistreamnum calibration position    buspos       device       sensor streamnum stream               path  start_s      end_s  duration_s  nsamples  avg_rate_hz        zones chanpos
15  i j k real    <4f  SGV                          l_thigh  spi0_106  DC_10000001       bno080         7  fquat  DC_10000001_fquat  0.03840  290.62690    290.5885     29073          100  b'\x04\x00'        
25  i j k real    <4f  SGV           1              l_shank        A1  SI_10000002  SI_10000002        35  fquat  SI_10000002_fquat  0.14595  290.09935    289.9534     29004          100  b'\x04\x00'   

```

Or selecting for stream 35 we can see what stream the 
```
>>> df[df['streamnum']=='35'][['fields','position','stream','path']]

        fields position stream               path
25  i j k real  l_shank  fquat  SI_10000002_fquat
```
** note `i j k real` fquat data gets converted to `x y z` euler angles internally for algos so `component 0 = x`

## Using Stream Data

The easiest way to extract streams from the npz for use is to use the helper function `tools.load_streams`.

```
>>> import cionic.tools as tools
>>> help(tools.load_streams

load_streams(npz, df=None, convert=True, degrees=False, **kwargs)
    Produces a list of streams from a given npz file
    
    Parameters: 
      npz:      npz file (required)
      df:       pandas dataframe of filtered segments (optional)
      convert:  convert 'emg' data to uV and 'fquat' data to euler
      degrees:  True to presnet eulers in degrees, false for radians
      **kwargs: filter npz segments by key value
    
    Returns:
      list of dictionaries matching filter criteria
                      [{ 'stream'   : (string),
                         'position' : (string),
                         'label'    : (string),
                         'segment'  : (string),
                         'values'   : (ndarray)
                      }]
```

There are two ways to filter down which streams to extract
1. supply a pandas DataFrame loaded from the `npz['segments']`
2. supply filter keyword arguments

Let's look first at the DataFrame method
```
>>> import pandas as pd
>>> import numpy as np
>>> from matplotlib import pyplot as plt
>>> npz = np.load('<path_to_npz>')                                 # load nps file
>>> segments = pd.DataFrame(npz['segments'])                       # create pandas DataFrame from segments
>>> segments = segments[segments['stream']=='fquat']               # filter to quaternion streams
>>> streams = tools.load_streams(npz, df=segments, convert=True)   # load streams converting quats to eulers
>>> plt.plot(streams[0]['values']['x'])
>>> plt.show()
```

<img width="622" alt="image" src="https://user-images.githubusercontent.com/264351/200649402-2679a87d-0c11-4d56-8b13-7d2a985828fd.png">

Now here the kwargs method for loading the same data
```
>>> npz = np.load('<path_to_npz>') 
>>> streams = tools.load_streams(npz, convert=True, stream='fquat')
```

## Displaying stim patterns

Starting from the example above we can add visualization of stim patterns for collections that include stimulation.
1. import the triggers module
2. specify the stim config using a dictionary with a key of DC_<serial>_action and the list of muscle numbers as an array
3. call `triggers.plot_stims_with_streams` passing the streams to graph and the stim_config (along with optional time bounds

```
>>> from cionic import triggers
>>> stim_config = {'DC_21000001_action': [1,2]}
>>> triggers.plot_stims_with_streams(npz, stim_config, streams, times=[200,203])
```

![stims](https://github.com/cionicwear/cionic-python/assets/264351/1746c108-fabd-4bce-ae07-86f23448ff62)



## Segmented NPZs

When a protocol contains labeled data, a segmented npz can be generated ([see README.md](README.md)]

The segmented npz segments each stream based on the timestamps of each label. 
If the protocol allows for a collection to have multiple segments with the same label, 
there will be a separate stream for each, differentiated by `segment_num`.
```
>>> import numpy as np
>>> import pandas as pd
>>> npz = np.load('<path_to_segmented_npz>')
>>> df = pd.DataFrame(npz['segments'])
>>> df[df['stream']=='emg'][['label','position','start_s','end_s','segment_num','path']]

         label     position     start_s       end_s  segment_num                 path
16  unassisted  l_shank_emg  250.551300  275.631012            1  SI_10000002_emg_001
17  stimulated  l_shank_emg  276.862488  306.021301            3  SI_10000002_emg_003
18      choice  l_shank_emg  245.788696  250.550797            0  SI_10000002_emg_000
19      choice  l_shank_emg  306.021790  308.061188            4  SI_10000002_emg_004
20      choice  l_shank_emg  275.631500  276.862000            2  SI_10000002_emg_002
```

Let's use the kwargs method to select `emg` streams for the `unassisted` and `assisted` labels.
```
>>> streams = tools.load_streams(npz, stream='emg', label=['unassisted','stimulated'])
>>> plt.plot(streams[0]['values']['c3'])
>>> plt.plot(streams[1]['values']['c3']
>>> plt.show()
```

<img width="587" alt="image" src="https://user-images.githubusercontent.com/264351/201162577-e633b24a-af47-4f7d-81f9-fbfc3b5236ba.png">







