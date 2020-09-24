import tensorflow_datasets as tfds
import numpy as np

def get_baseline_data(data_type, y_label, subsample=200):
    uppermax = int(subsample*20)
    ds = tfds.load(data_type, split='train').prefetch(1)
    dslist = np.array(list(tfds.as_numpy(ds)))
    nrows = len(dslist)
    m = np.random.choice(nrows, uppermax, replace=False)
    dslist = dslist[m]
    res = []
    for label in y_label:
        base_img_all = []
        for ds in dslist:
            if ds.get("label") == label:
                base_img_all += [ds.get("image")]
        base_img_all = np.array(base_img_all)
        N = base_img_all.shape[0]
        m = np.random.choice(N, subsample, replace=False)
        res +=[ base_img_all[m]]
    return np.array(res)

if __name__ == '__main__':
    base = get_baseline_data("mnist", 0, subsample=200)
    print("final")
