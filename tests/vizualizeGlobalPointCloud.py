
from pathlib import Path

import common.FGVizHelpers as fgtools
import open3d as o3d
import matplotlib.pyplot as plt
import json
import gtsam

from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--directory", type=str, nargs='+', default=None, help="Specify which session directory to load")
    parser.add_argument("-s", "--save", action='store_true', help="Flag to save the final point cloud")
    parser.add_argument("-v", "--voxel-size", type=float, default=0.5, help="Voxel size for global map")
    args = parser.parse_args()

    # Print as a clean, structured list
    # print("------------------------------")
    print("\n--- Command Line Arguments ---")
    # print("------------------------------")
    # for key, value in vars(args).items():
    #     print(f"{key}: {value}")
    # print("------------------------------")

    print(json.dumps(vars(args), indent=2))


    savePointCloud = args.save

    if not args.directory:
        sessionDir = Path("/home/benb/blocklio/python/results/nc/post/park__cloister__quad-hard__quad-medium__quad-easy/INLIER_RATIO")

    for sessionDir in args.directory:
        directory = Path(sessionDir).absolute()
        print(f"\nCreating global colored cloud for {str(directory)}")
        print("Loading session...")

        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory {directory} does not exist.")
        
        odomFile = directory / "odom_tum.txt"
        timestampsFile = directory / "timestamps.txt"

        if Path(directory.parent / "scans").exists():
            scansDir = directory.parent / "scans"
        elif Path(directory / "scans").exists():
            scansDir = directory / "scans"
        else:
            raise ValueError(f"LoadSession: Can't find 'scans' directory in directory or its parent - {directory}")

        key2timestamps = fgtools.loadTimestampsWithKeys(str(timestampsFile))
        timestamps2keys = {value: key for key, value in key2timestamps.items()}

        sharedScanSymbols = fgtools.discoverScanSymbols(scansDir)
        
        pcds = fgtools.loadSymbolPCDs(scansDir, keys=list(key2timestamps.keys()))
        if not pcds:
            raise ValueError("LoadSession: pcds after loading is still empty!")

        timestamp2pose = fgtools.loadOdomTUM(odomFile)

        values = gtsam.Values()

        for t, p in timestamp2pose.items():
            key = timestamps2keys[t]
            values.insert(key, p)
        
        print("Plotting graph points...")
        fgtools.plot_points(values.keys(), values, changeColorBySymbols=True)

        plt.pause(2)

        print("Creating point cloud...")
        globalPCD = fgtools.createGlobalPCD(pcds, values, voxel_size=None if args.voxel_size == 0.0 else args.voxel_size, changeColorBySymbols=True)

        o3d.visualization.draw_geometries([globalPCD])

        if savePointCloud:
            pcdFile = sessionDir / "globalMapColored.pcd"
            print(f"Saving colored global map to {str(pcdFile)}")
            o3d.io.write_point_cloud(pcdFile, globalPCD)
        print("Done")



