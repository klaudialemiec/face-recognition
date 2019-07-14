import argparse
from experiments import experiment, experiment_camera, experiment_add_person


def start_demo(exp_number):
    print('[INFO] Starting demo: ', exp_number)
    if exp_number == 1:
        experiment.launch_standard_experiment()
    elif exp_number == 2:
        experiment.launch_modified_experiment('grayscale')
    elif exp_number == 3:
        experiment.launch_modified_experiment('blur')
    elif exp_number == 4:
        experiment.launch_modified_experiment('equalize')
    elif exp_number == 5:
        experiment.launch_modified_experiment('gamma')
    elif exp_number == 6:
        experiment.launch_modified_experiment('none')
    elif exp_number == 7:
        experiment.launch_modified_experiment('gamma_grayscale')
    elif exp_number == 8:
        experiment.launch_modified_experiment('histogram_grayscale')


def start_live(source, path):
    print('[INFO] Starting live, image source: ', source, path)
    if source == 'file' and path is not None:
        pass

    elif source == 'camera':
        experiment_camera.launch_camera_experiment()
    else:
        print("[ERROR] Path to JPG file required.")


def start_add(source, name, path):
    print('[INFO] Starting adding to base, image source: ', source, path)
    if source == 'file' and path is not None:
        experiment_add_person.launch_add_person_experiment_file(name, path)
    elif source == 'camera':
        experiment_add_person.launch_add_person_experiment_camera(name)
    else:
        print("[ERROR] Something went wrong.")


def prepare_parser():
    parser = argparse.ArgumentParser(description='Face recognition program. Please choose required mode when launching.')
    parser.add_argument('-mode',
                        choices=['live', 'demo', 'add'],
                        required=True,
                        help='Demo - launch one of prepared experiments, requires -exp (1-3) argument \n'
                             'Live - recognize face from file or camera \n'
                             'Add - add person to base from from file or camera')
    parser.add_argument('-exp',
                        type=int,
                        choices=[1, 2, 3, 4, 5, 6, 7, 8],
                        help='Experiment scenario number (1-8).')
    parser.add_argument('-source',
                        choices=['camera', 'file'],
                        help='Source of face image. Required if -mode=live. Can be either camera of your PC or existing JPG file. If set to file -path argument required')
    parser.add_argument('-path',
                        type=str,
                        help='Path to JPG file. Required if -mode=live and -source=file')
    parser.add_argument('-name',
                        type=str,
                        help='Name of person whos face is added to base. Required if -mode=add')
    return parser


if __name__ == '__main__':
    parser = prepare_parser()
    args = parser.parse_args(['-mode', 'demo', '-exp', '6'])
    if args.mode == 'live':
        if args.source is not None:
            start_live(args.source, args.path)
        else:
            print("[ERROR] Image source required. Please set -source to 'camera' or 'file'")
    elif args.mode == 'demo':
        if args.exp is not None:
            start_demo(args.exp)
        else:
            print("[ERROR] Experiment number required. Please set -exp to 1, 2 or 3")
    elif args.mode == 'add':
        if args.source is not None and args.name is not None:
            start_add(args.source, args.name, args.path)
        else:
            print("[ERROR] Image source and name required. Please set -source to 'camera' or 'file' and provide name")
