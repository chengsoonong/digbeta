#!/usr/bin/env python3
import requests
import tempfile
import zipfile
import os
import sys

def download_data(url):
    """Download data from a specified URL, save it to a temp file, 
       return the full path of the temp file.
    """
    print('Downloading data from', url)
    fname = None
    r = requests.get(url, stream=True)
    with tempfile.NamedTemporaryFile(delete=False) as fd:
        fname = fd.name
        for chunk in r.iter_content(chunk_size):
            fd.write(chunk)
    return fname


def unzip_file(fname, path):
    """Unzip a zipfile to a specified path"""
    assert(zipfile.is_zipfile(fname))
    assert(os.path.exists(path))
    zipfile.ZipFile(fname).extractall(path=path)
    print('Data available in directory', path)


if __name__ == '__main__':
    data_url = 'https://sites.google.com/site/limkwanhui/datacode/data-ijcai15.zip?attredirects=0'
    coord_url = 'https://www.dropbox.com/s/nd4o5u3nwrepjqq/data-ijcai15-coords.zip?dl=1'
    chunk_size = 4096
    data_dir = '../data'
    subdir = 'data-ijcai15'

    if os.path.exists(data_dir) == False:
        print('Data directory "' + data_dir + '" not found,')
        print('Please create it.')
        sys.exit(0)

    # download/unzip part1 of data
    fname1 = download_data(data_url)
    unzip_file(fname1, data_dir)

    # download/unzip part2 of data
    fname2 = download_data(coord_url)
    unzip_file(fname2, os.path.join(data_dir, subdir))

    # delete temp files
    os.unlink(fname1)
    os.unlink(fname2)

