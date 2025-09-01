import argparse, zipfile, os, sys, urllib.request, shutil
URL = "https://www.timeseriesclassification.com/Downloads/ECG5000.zip"
def download(url, out_zip):
    with urllib.request.urlopen(url) as r, open(out_zip, "wb") as f:
        shutil.copyfileobj(r, f)
def extract(zip_path, out_dir):
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/ECG5000")
    ap.add_argument("--keep_zip", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    zip_path = os.path.join(args.out, "ECG5000.zip")
    if not os.path.exists(zip_path):
        download(URL, zip_path)
    extract(zip_path, args.out)
    if not args.keep_zip:
        os.remove(zip_path)
