import os


NETCDF_EXTENSION = [
    '.nc4'
]


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_netcdf_dataset(dir):
    cells = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_netcdf_file(fname):
                path = os.path.join(root, fname)
                cells.append(path)

    return cells


def is_netcdf_file(filename):
    return any(filename.endswith(extension) for extension in NETCDF_EXTENSION)