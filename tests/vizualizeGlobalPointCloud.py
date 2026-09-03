
from pathlib import Path

from multiSession.SessionManager import SessionManager
import common.FGVizHelpers as fgtools
import open3d as o3d
import matplotlib.pyplot as plt
import json

from argparse import ArgumentParser

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
    sessionDir = Path(sessionDir).absolute()
    print(f"\nCreating global colored cloud for {str(sessionDir)}")
    print("Loading session...")
    session = SessionManager(sessionDir)

    print("Plotting graph...")
    fgtools.plot_graph(session.graph, session.values, changeColorBySymbols=True)

    plt.pause(2)

    print("Creating point cloud...")
    globalPCD = session.CreateGlobalPCD(voxel_size=None if args.voxel_size == 0.0 else args.voxel_size, changeColorBySymbols=True)

    o3d.visualization.draw_geometries([globalPCD])

    if savePointCloud:
        pcdFile = sessionDir / "globalMapColored.pcd"
        print(f"Saving colored global map to {str(pcdFile)}")
        o3d.io.write_point_cloud(pcdFile, globalPCD)
    print("Done")



