from pathlib import Path


def getMissionPaths(dataset: str, mission: str) -> tuple[Path, Path]:
    '''
    Return (bag_file, gt_file) for a mission in a dataset.

    :param dataset: One of "nc", "nclt", or "urban"
    :param mission: Mission name within the dataset (e.g. "quad-easy", "2012-01-08", "block0")
    '''
    dataset = dataset.lower()
    if dataset == "nc":
        nc_data_dir = Path.home() / "data_sets/NewerCollege/ros2"
        bag_file = nc_data_dir / mission
        gt_file = nc_data_dir / "ground_truth_tum" / f"gt-nc-{mission}.csv"

    elif dataset == "urban":
        urban_data_dir = Path.home() / "data_byu/bags/urban"
        bag_file = urban_data_dir / mission
        gt_file = urban_data_dir / "groundTruth" / f"{mission}_gt" / "optimized_poses.txt"

    elif dataset == "nclt":
        nclt_data_dir = Path.home() / "data_sets/NCLT/nclt2ros/"
        bag_file = nclt_data_dir / f"rosbags/{mission}/ros2_bag/"
        gt_file = nclt_data_dir / f"raw_data/{mission}/ground_truth/groundtruth_{mission}.csv"

    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Options: nc, nclt, urban")

    return bag_file, gt_file


def isKnownMission(dataset: str, mission: str) -> bool:
    '''Return True if the mission's bag exists for the dataset.'''
    try:
        bag_file, _ = getMissionPaths(dataset, mission)
    except ValueError:
        return False
    return bool(mission) and bag_file.exists()
