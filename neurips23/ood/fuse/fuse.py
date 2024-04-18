from __future__ import absolute_import
import psutil
import os
import time
import numpy as np
import pyanns
import diskannpy
import gc
import index_mystery
from urllib.request import urlopen

from neurips23.ood.base import BaseOODANN
from benchmark.datasets import DATASETS, download_accelerated

def download_train(src, dst=None, max_size=None):
        """ download an URL, possibly cropped """
        # if os.path.exists(dst):
        #     return
        print('downloading %s -> %s...' % (src, dst))
        if max_size is not None:
            print("   stopping at %d bytes" % max_size)
        t0 = time.time()
        outf = open(dst, "wb")
        inf = urlopen(src)
        info = dict(inf.info())
        content_size = int(info['Content-Length'])
        bs = 1 << 20
        totsz = 0
        while True:
            block = inf.read(bs)
            elapsed = time.time() - t0
            print(
                "  [%.2f s] downloaded %.2f MiB / %.2f MiB at %.2f MiB/s   " % (
                    elapsed,
                    totsz / 2**20, content_size / 2**20,
                    totsz / 2**20 / elapsed),
                flush=True, end="\r"
            )
            if not block:
                break
            if max_size is not None and totsz + len(block) >= max_size:
                block = block[:max_size - totsz]
                outf.write(block)
                totsz += len(block)
                break
            outf.write(block)
            totsz += len(block)
        print()
        print("download finished in %.2f s, total size %d bytes" % (
            time.time() - t0, totsz
        ))

class Fuse(BaseOODANN):
    def __init__(self, metric, index_params):
        self.name = "Fuse"

        self._index_params = index_params
        self._metric = metric

        self.R = index_params.get("R")
        self.L = index_params.get("L")
        if (index_params.get("M_pjbp")==None):
            print("Error: missing parameter M_pjbp")
            return
        if (index_params.get("L_pjpq")==None):
            print("Error: missing parameter L_pjpq")
            return
        if (index_params.get("NoP")==None):
            print("Error: missing parameter NoP")
            return
        if (index_params.get("T")==None):
            print("Error: missing parameter T for set threads")
            return
        if (index_params.get("NoT")==None):
            print("Error: missing parameter NoT")
            return
        if (index_params.get("EoP")==None):
            print("Error: missing parameter EoP")
            return
        
        self.M_pjbp = index_params.get("M_pjbp")
        self.L_pjpq = index_params.get("L_pjpq")
        self.NoT = index_params.get("NoT")
        self.EoP = index_params.get("EoP")
        self.NoP = index_params.get("NoP")
        self.T = index_params.get("T")

        self.dir = "indices"
        self.path = self.index_name()

    def index_name(self):
        return f"M_bp{self.M_pjbp}_L_pq{self.L_pjpq}_NoT{self.NoT}_ord"
            
    def create_index_dir(self, dataset):
        index_dir = os.path.join(os.getcwd(), "data", "indices", "ood")
        os.makedirs(index_dir, mode=0o777, exist_ok=True)
        index_dir = os.path.join(index_dir, 'diskann')
        os.makedirs(index_dir, mode=0o777, exist_ok=True)
        index_dir = os.path.join(index_dir, dataset.short_name())
        os.makedirs(index_dir, mode=0o777, exist_ok=True)
        index_dir = os.path.join(index_dir, self.index_name())
        os.makedirs(index_dir, mode=0o777, exist_ok=True)
        return index_dir


    def translate_dist_fn(self, metric):
        if metric == 'euclidean':
            return index_mystery.L2
        elif metric == 'ip':
            return index_mystery.IP
        elif metric == 'ip_build':
            return index_mystery.IP_BUILD
        elif metric == 'cosine':
            return index_mystery.COSINE
        else:
            raise Exception('Invalid metric')

    def translate_dist_fn_pyanns(self, metric):
        if metric == 'euclidean':
            return 'L2'
        elif metric == 'ip':
            return 'IP'
        else:
            raise Exception('Invalid metric')

    def translate_dtype(self, dtype: str):
        if dtype == 'uint8':
            return np.uint8
        elif dtype == 'int8':
            return np.int8
        elif dtype == 'float32':
            return np.float32
        else:
            raise Exception('Invalid data type')

    def fit(self, dataset):
        """
        Build the index for the data points given in dataset name.
        """
        ds = DATASETS[dataset]()
        d = ds.d

        buildthreads = self._index_params.get("buildthreads", -1)
        print(buildthreads)
        if buildthreads == -1:
            buildthreads = 0

        index_dir = self.create_index_dir(ds)
        save_name = f'learn.{self.NoT}M.fbin'
        # save_path = os.path.join(ds.basedir, save_name)
        save_path = os.path.join(index_dir, save_name)

        max_size = self.NoT * 1000000 * 200 * 4 + 8
        src_url = 'https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.learn.50M.fbin'
        print(src_url, save_path, max_size)
        
        if ds.d == 200:
            download_train(src_url, save_path, max_size)

            with open(save_path, 'rb+') as f:
                npts = self.NoT * 1000000
                f.write(int(npts).to_bytes(4, byteorder='little'))
                f.close()
        
        if  hasattr(self, 'index'):
            print('Index object exists already')
            return

        print(ds.get_dataset_fn())
        
        graph_path = f'{index_dir}/{self.index_name()}'
        
        if not os.path.exists(graph_path):
            start = time.time()
            index_mystery.buildST2(self.M_pjbp, self.L_pjpq, self.T, os.path.join(os.getcwd(), ds.get_dataset_fn()), self.translate_dist_fn(str(ds.distance())), save_path, self.EoP, self.NoT, index_dir)
            end = time.time()
            print("Fuse index built in %.3f s" % (end - start))
        gc.collect()
        g = pyanns.Graph(graph_path, 'roargraph')
        self.searcher = pyanns.Searcher(
            g, ds.get_dataset(), self.translate_dist_fn_pyanns(ds.distance()), "SQ8U")
        self.searcher.optimize()
        print('Index ready for search')

    def load_index(self, dataset):
        """
        Load the index for dataset. Returns False if index
        is not available, True otherwise.

        Checking the index usually involves the dataset name
        and the index build paramters passed during construction.
        """
        return False

    def query(self, X, k):
        """Carry out a batch query for k-NN of query set X."""
        nq, _ = X.shape
        self.res = self.searcher.batch_search(X, k).reshape(nq, -1)

    def set_query_arguments(self, query_args):
        self.ef = query_args.get("ef")
        self.searcher.set_ef(self.ef)

    def __str__(self):
        return f'fuse({self.index_name()})_ef{self.ef}'