#!/usr/bin/env python3

"""
Generate activation files

Generate activation files from raw extracted data
"""

import argparse
import numpy as np
import os
import pickle


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
                audio_id.append(k + '.wav')
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
            for lname in layers:
                if lname not in activations:
                    activations[lname] = []
                ac = v['activations'][lname].squeeze()
                activations[lname].append(ac)

        for lname in layers:
            # Saving splitted activation data
            fout_path = fpath.rstrip('.pkl') + '.{}.tmp'.format(lname)
            pickle.dump(activations[lname], open(fout_path, 'wb'), protocol=4)

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

    for lname in layers:
        activations = []
        for fpath in flist:
            fpart_path = fpath.rstrip('.pkl') + '.{}.tmp'.format(lname)
            with open(fpart_path, 'rb') as f:
                data = pickle.load(f)
            activations.extend(data)

        print('Number of frames (' + lname + '):',
              np.sum([ac.shape[0] for ac in activations]))
        fout_name = 'global_activations.{}.pkl'.format(lname)
        fout_path = os.path.join(root, fout_name)
        pickle.dump(np.array(activations), open(fout_path, 'wb'), protocol=4)



def main(num_splits):
    layers = ['convout']
    layers.extend(['transf{}'.format(i) for i in range(0, 12)])

    fpath = [('exp/train_960_pytorch_phonemerepr',
              'decode_dev_{}_model.val5.avg.best_decode_pytorch_transformer_large_'),
             ('exp/train_960_pytorch_phonemerepr',
              'decode_dev_{}_model.acc.init_decode_pytorch_transformer_large_')]
    for root, folder_template in fpath:
        # Generating list of files to combine
        flist = []
        for subset in ['clean', 'other']:
            folder = folder_template.format(subset)
            for i in range(1, num_splits + 1):
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


if __name__ == '__main__':
    # Parsing command line
    doc = __doc__.strip("\n").split("\n", 1)
    parser = argparse.ArgumentParser(
        description=doc[0], epilog=doc[1],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-n', '--num_splits', help='Number of splits', type=int, default=None)
    args = parser.parse_args()

    main(args.num_splits)
