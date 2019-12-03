import numpy as np
import os
import pickle


num_layers_encoder = 12
num_split = 8


def load_metadata(flist):
    # Initializing arrays
    audio_id = []
    text = []
    audio = []

    for fpath in flist:
        # Filling arrays
        with open(fpath, 'rb') as f:
            data = pickle.load(f)

            for k, v in data.items():
                audio_id.append(k)
                text.append(v['text'])
                audio.append(v['input'])

    # Converting to numpy arrays
    metadata = {'audio_id': np.array(audio_id),
                'text': np.array(text),
                'audio': np.array(audio)}
    return metadata


def load_metadata_and_split(flist, layers, valid_ids):
    # Initializing arrays
    audio_id = []
    text = []
    audio = []

    for fpath in flist:
        activations = {}
        with open(fpath, 'rb') as f:
            data = pickle.load(f)

        for k, v in data.items():
            if k not in valid_ids:
                continue
            # Filling arrays
            audio_id.append(k)
            text.append(v['text'])
            audio.append(v['input'])

            # Loading activation data
            for l in layers:
                if l['name'] not in activations:
                    activations[l['name']] = []
                ac = l['access_fn'](v['activations']).squeeze()
                activations[l['name']].append(ac)

        for l in layers:
            # Saving splitted activation data
            fout_path = fpath.rstrip('.pkl') + '.{}.tmp'.format(l['name'])
            pickle.dump(activations[l['name']], open(fout_path, 'wb'), protocol=4)

    # Converting to numpy arrays
    metadata = {'audio_id': np.array(audio_id),
                'text': np.array(text),
                'audio': np.array(audio)}
    return metadata


def load_activations(access_fn, flist):
    # Initialization
    activations = []

    for fpath in flist:
        # Filling arrays
        with open(fpath, 'rb') as f:
            data = pickle.load(f)
            for v in data.values():
                activations.append(access_fn(v['activations']).squeeze())

    # Converting to numpy arrays
    activations = np.array(activations)
    return activations


def merge_activations(root, flist, layers):
    # Initialization

    for l in layers:
        activations = []
        for fpath in flist:
            fpart_path = fpath.rstrip('.pkl') + '.{}.tmp'.format(l['name'])
            with open(fpart_path, 'rb') as f:
                data = pickle.load(f)
            activations.extend(data)

        print('Number of frames (' + l['name'] + '):',
              np.sum([ac.shape[0] for ac in activations]))
        fout_name = 'global_activations.{}.pkl'.format(l['name'])
        fout_path = os.path.join(root, fout_name)
        pickle.dump(np.array(activations), open(fout_path, 'wb'), protocol=4)


layers = [{'name': 'conv1', 'access_fn': lambda x: x['embed']['conv1']},
          {'name': 'conv2', 'access_fn': lambda x: x['embed']['conv3']},
          {'name': 'convout', 'access_fn': lambda x: x['embed']['out']}]
for i in range(0, num_layers_encoder):
    src_name = 'layer' + str(i)
    tgt_name = 'transf' + str(i)
    layers.append({'name': tgt_name,
                   'access_fn': lambda x, sn=src_name: x['encoders'][sn]})

fpath = [('exp/train_960_pytorch_trained',
          'decode_dev_{}_model.val5.avg.best_decode_pytorch_transformer_large_'),
         ('exp/train_960_pytorch_random',
          'decode_dev_{}_model.acc.init_decode_pytorch_transformer_large_')]
for root, folder_template in fpath:
    # Generating list of files to combine
    flist = []
    for subset in ['clean', 'other']:
        folder = folder_template.format(subset)
        for i in range(1, num_split + 1):
            fname = 'data.' + str(i) + '.pkl'
            flist.append(os.path.join(root, folder, fname))

    # Loading list of valid IDS
    with open('val_valid_ids.txt') as f:
        valid_ids = [l.rstrip('.wav\n') for l in f.readlines()]

    # Generating metadata
    metadata = load_metadata_and_split(flist, layers, valid_ids)
    print('Number of audio files:', metadata['audio_id'].shape[0])
    print('Number of frames (input):',
          np.sum([inp.shape[0] for inp in metadata['audio']]))
    ffullpath = os.path.join(root, 'global_input.pkl')
    pickle.dump(metadata, open(ffullpath, 'wb'), protocol=4)

    # Generating activation files layer by layer
    merge_activations(root, flist, layers)
    #for l in layers:
    #    activations = load_activations(l['access_fn'], flist)
    #    # Printing some stats
    #    print('Number of frames (' + l['name'] + '):',
    #          np.sum([ac.shape[0] for ac in activations]))
    #    # Saving data
    #    ffullpath = os.path.join(
    #        root, 'global_activations.' + l['name'] + '.pkl')
    #    pickle.dump(activations, open(ffullpath, 'wb'), protocol=4)
