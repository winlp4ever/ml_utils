# a small script to handle downloading data from urls
import urllib
from urllib.request import urlretrieve
from urllib.parse import urlparse
import os, sys
import zipfile
import math
import time


def error_handle():
    """Print out error line and filename
    (from stack overflow)
    """
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(exc_type, fname, exc_tb.tb_lineno)


def retri_file_size(url: str):
    """Return to-be-downloaded file size
    """
    meta = urllib.request.urlopen(url).info()
    return int(meta['Content-Length'])


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
    def indicator(quantity, width=12):
        if quantity > 1024:
            return '{:.0f} MB/s'.format(quantity / 1024).rjust(width)
        return '{:.0f} KB/s'.format(quantity).rjust(width)

    def progress(count, block_size, total_size):
        global start_time
        if count == 0:
            start_time = time.time()
        if count % 100 == 99 or (count + 1) * block_size >= total_size:
            percent = (count * block_size) / total_size * 100.
            pos = int(math.ceil(percent / 100 * 20))
            down_bar = '[' + '=' * max(pos - 1, 0) + '>' + (20 - pos) * '-' + ']'

            print('{} {}/{} MB {} {}'.format(down_bar,
                    str(min((count + 1) * block_size, total_size) // (1024 * 1024)).rjust(6), # right align text
                    total_size // (1024 * 1024), ('(%2.1f%%)'%percent).rjust(8),
                    indicator((count * block_size) / (time.time() - start_time + 1e-3) / 1024)),
                flush=True, end='\r')
    for url in urls:
        try:
            fn = retri_fn_url(url)
            save_path = os.path.join(save_dir, fn)
            if os.path.exists(save_path) and os.path.getsize(save_path) >= retri_file_size(url):
                print('{} already exists.'.format(save_path))
                continue
            print('Downloading {} ...'.format(fn))
            urlretrieve(url, save_path, reporthook=progress)
            print('\n')
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
    urls = ['https://www.dropbox.com/s/h4pypk9s2mxzzme/checkpoint-3.pth.tar?dl=1']
    down_fr_url(urls, unzip=False)
