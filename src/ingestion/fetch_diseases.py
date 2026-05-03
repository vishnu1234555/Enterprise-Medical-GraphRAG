import ftplib
import os

FTP_HOST = "ftp.ebi.ac.uk"
BASE_DIR = "/pub/databases/opentargets/platform/"
LOCAL_DIR = "data/diseases/"


def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)

    print(f"Connecting to {FTP_HOST}...")
    ftp = ftplib.FTP(FTP_HOST)
    ftp.login()  # Anonymous

    print("Finding the latest release version...")
    ftp.cwd(BASE_DIR)
    folders = []
    ftp.retrlines("LIST", lambda line: folders.append(line.split()[-1]))

    versions = sorted([f for f in folders if "." in f and f[0].isdigit()], reverse=True)

    if not versions:
        print("[FATAL] Could not find version directories.")
        return

    latest_version = versions[0]
    print(f"Targeting OpenTargets Release: {latest_version}")

    possible_paths = [
        f"{BASE_DIR}{latest_version}/output/disease/",
        f"{BASE_DIR}{latest_version}/output/etl/parquet/diseases/",
    ]

    target_dir = None
    for path in possible_paths:
        try:
            ftp.cwd(path)
            target_dir = path
            break
        except ftplib.error_perm:
            continue

    if not target_dir:
        print(f"[FATAL] Could not find the diseases folder in release {latest_version}")
        return

    print(f"Located dictionary directory: {target_dir}")

    files = ftp.nlst()
    parquet_files = [f for f in files if f.endswith(".parquet")]

    if not parquet_files:
        print("[FATAL] No parquet files found in the directory.")
        return

    print(f"Found {len(parquet_files)} dictionary files. Beginning download...")

    for filename in parquet_files:
        local_path = os.path.join(LOCAL_DIR, filename)
        if os.path.exists(local_path):
            print(f"Skipping {filename} (Already exists)")
            continue

        print(f"Downloading {filename}...")
        with open(local_path, "wb") as f:
            ftp.retrbinary(f"RETR {filename}", f.write)

    print("[SUCCESS] All disease dictionaries downloaded.")
    ftp.quit()


if __name__ == "__main__":
    main()
