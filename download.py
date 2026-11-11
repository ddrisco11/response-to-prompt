from huggingface_hub import snapshot_download
from settings import BASE, BASE_REV

if __name__ == '__main__':
    snapshot_download(BASE, revision=BASE_REV, local_dir='.assets/base')
