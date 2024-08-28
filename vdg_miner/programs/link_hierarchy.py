import os
import sys

def create_symlinks_for_pdb_files(root_path):
    """
    Traverse the directory tree and create symlinks for .pdb files in each parent directory,
    except the directory that contains the .pdb file.
    """
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.endswith('.pdb'):
                pdb_file_path = os.path.join(dirpath, filename)
                parent_path = dirpath

                # Traverse each parent directory except the one containing the .pdb file
                while parent_path != root_path:
                    parent_path = os.path.dirname(parent_path)
                    symlink_path = os.path.join(parent_path, filename)

                    if not os.path.exists(symlink_path):
                        try:
                            os.symlink(pdb_file_path, symlink_path)
                            # print(f"Created symlink: {symlink_path} -> {pdb_file_path}")
                        except Exception as e:
                            print(f"Failed to create symlink: {symlink_path} -> {pdb_file_path}, due to: {e}")

def count_non_directory_files(folder_path):
    """
    Count the number of non-directory files in a given folder.
    """
    count = 0
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            count += 1
    return count

def rename_folders_with_file_count(root_path):
    """
    Traverse the directory tree and rename each folder to include the count of non-directory files.
    """
    for dirpath, dirnames, filenames in os.walk(root_path, topdown=False):
        # Skip the root directory itself
        if dirpath == root_path:
            continue

        # Get the current folder name and parent path
        current_folder_name = os.path.basename(dirpath)
        parent_path = os.path.dirname(dirpath)

        # If "rescount" in current folder name, strip it away
        if "rescount" in current_folder_name:
            current_folder_name = '_'.join(current_folder_name.split('_')[:-2])

        # Count the non-directory files in the current folder
        file_count = count_non_directory_files(dirpath)

        # Construct the new folder name
        new_folder_name = f"{current_folder_name}_rescount_{file_count}"

        # Construct the full new path
        new_folder_path = os.path.join(parent_path, new_folder_name)

        # Rename the folder
        os.rename(dirpath, new_folder_path)

        print(f"Renamed: {dirpath} -> {new_folder_path}")

def count_files_and_rename_dirs_at_depth(starting_dir, target_depth):
    """
    Traverses the directory tree starting from `starting_dir`, counts the
    number of non-directory files in each sub-tree at a specified depth,
    and renames each directory at that depth to include the count of
    non-directory files.

    :param starting_dir: The root directory from which to start the traversal.
    :param target_depth: The depth below which directories should be renamed.
    """
    def get_depth(path):
        return path[len(starting_dir):].count(os.sep)

    for root, dirs, files in os.walk(starting_dir, topdown=False):
        current_depth = get_depth(root)

        if current_depth >= target_depth:
            # Count the number of non-directory files in the current directory
            # and its subdirectories
            num_files = sum([len(files) for _, _, files in os.walk(root)])

            # Get the new directory name with the count of non-directory files
            base_dir = os.path.basename(root)
            parent_dir = os.path.dirname(root)
            new_dir_name = f"{base_dir}_rescount_{num_files}"
            new_dir_path = os.path.join(parent_dir, new_dir_name)

            # Rename the directory
            os.rename(root, new_dir_path)

assert len(sys.argv) > 1
hierarchy_dir = os.path.realpath(sys.argv[1])

count_files_and_rename_dirs_at_depth(hierarchy_dir, 1)
create_symlinks_for_pdb_files(hierarchy_dir)
