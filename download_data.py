# a small script to handle downloading data from urls

from urllib.request import urlretrieve
from urllib.parse import urlparse
import os, sys
import zipfile

def error_handle():
    """
    print out error line and filename
    (from stack overflow)
    """
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(exc_type, fname, exc_tb.tb_lineno)


def retri_fn_url(url: str) -> str:
    """return filename from an url
    Parameters
    ----------
    url : str
        link url
    Returns
    -------
    str
        url file name
    """
    addr = urlparse(url).path
    return os.path.basename(addr)


def down_fr_url(urls: list, save_dir: str='', unzip: bool=False):
    for url in urls:
        try:
            fn = retri_fn_url(url)
            save_path = os.path.join(save_dir, fn)
            print('Downloading ... {}'.format(fn))
            urlretrieve(url, save_path)
            if unzip:
                zip = zipfile.ZipFile(save_path)
                zip.extractall('.')
                zip.close()

        except Exception as e:
            error_handle()
            print(e)
    print('Done.')


if __name__ == '__main__':
    test = 'funny'
    print(retri_fn_url(test))
    urls = ['']
    down_fr_url(urls, unzip=True)
