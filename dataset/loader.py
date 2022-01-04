from dataset.ucf101 import UcfHmdb
from pprint import pprint

def get_training_set(args):
    assert args.dataset in ['ucf101', 'hmdb51']

    if args.dataset == 'ucf101':
        training_data = UcfHmdb(
            'UCF101',
            args.video_path,
            args.dataset_file,
            'training',
            testing=False,
            num_frames=args.sample_duration // args.stride_size,
            sample_stride=args.stride_size,
            n_samples_for_each_video=1,
            n_samples_for_each_frame=1,
            crop_size=args.sample_size)
    elif args.dataset == 'hmdb51':
        training_data = UcfHmdb(
            'HMDB51',
            args.video_path,
            args.dataset_file,
            'training',
            testing=False,
            num_frames=args.sample_duration // args.stride_size,
            sample_stride=args.stride_size,
            n_samples_for_each_video=1,
            n_samples_for_each_frame=1,
            crop_size=args.sample_size)

    return training_data


def get_validation_set(args):
    assert args.dataset in ['ucf101', 'hmdb51']

    if args.dataset == 'ucf101':
        validation_data = UcfHmdb(
            'UCF101',
            args.video_path,
            args.dataset_file,
            'validation',
            testing=False,
            num_frames=args.sample_duration // args.stride_size,
            sample_stride=args.stride_size,
            n_samples_for_each_video=1,
            n_samples_for_each_frame=1,
            crop_size=args.sample_size)
    elif args.dataset == 'hmdb51':
        validation_data = UcfHmdb(
            'HMDB51',
            args.video_path,
            args.dataset_file,
            'validation',
            testing=False,
            num_frames=args.sample_duration // args.stride_size,
            sample_stride=args.stride_size,
            n_samples_for_each_video=1,
            n_samples_for_each_frame=1,
            crop_size=args.sample_size)

    return validation_data


def get_test_set(args):
    assert args.dataset in ['ucf101', 'hmdb51']
    assert args.test_subset in ['val', 'test']

    if args.test_subset == 'val':
        subset = 'validation'
    elif args.test_subset == 'test':
        subset = 'testing'

    if args.dataset == 'ucf101':
        test_data = UcfHmdb(
            'UCF101',
            args.video_path,
            args.dataset_file,
            subset,
            testing=True,
            num_frames=args.sample_duration // args.stride_size,
            sample_stride=args.stride_size,
            n_samples_for_each_video=10,
            n_samples_for_each_frame=3,
            crop_size=args.sample_size)
    elif args.dataset == 'hmdb51':
        test_data = UcfHmdb(
            'HMDB51',
            args.video_path,
            args.dataset_file,
            subset,
            testing=True,
            num_frames=args.sample_duration // args.stride_size,
            sample_stride=args.stride_size,
            n_samples_for_each_video=10,
            n_samples_for_each_frame=3,
            crop_size=args.sample_size)

    return test_data