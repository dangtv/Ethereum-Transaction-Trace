# %%
import os

def get_file_sizes(dir_path):
    """
    Recursively get the sizes of all files in the directory and its subdirectories.
    """
    file_sizes = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                size = os.path.getsize(file_path)
                file_sizes.append((file_path, size))
            except Exception as e:
                print(f"Error getting size for file {file_path}: {e}")
    return file_sizes

def main():
    path = "./../TracingData4"
    file_sizes = get_file_sizes(path)
    file_sizes.sort(key=lambda x: x[1], reverse=True)  # Sort by size in descending order

    # Print the sorted file sizes
    for file_path, size in file_sizes:
        if size > 100 * 1024 * 1024:
            print(f"{file_path}: {size / (1024 * 1024)} Mb")

if __name__ == "__main__":
    main()


# %%
import os

def get_and_delete_large_files(dir_path, size_threshold):
    """
    Recursively get and delete files larger than the specified size threshold.
    """
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                size = os.path.getsize(file_path)
                if size > size_threshold:
                    os.remove(file_path)
                    print(f"Deleted {file_path}: {size} bytes")
            except Exception as e:
                print(f"Error handling file {file_path}: {e}")

def main():
    path = "./../TracingData4"
    size_threshold = 100 * 1024 * 1024  # Set size threshold to 200 MB 
    
    get_and_delete_large_files(path, size_threshold)

if __name__ == "__main__":
    main()



