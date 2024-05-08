from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-g", "--gs_name", type=str, help="Unique grid search name to aggregate")
parser.add_argument("-d", "--deer", type=str, help="deer stuff")
args = parser.parse_args()
gs_name = args.gs_name
deer_name = args.deer

if any([val is not None for val in args.__dict__.values()]):
    print('cli mode')
else:
    print('not cli mode')